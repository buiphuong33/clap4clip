# Giải thích mô hình CLAP4CLIP bằng tiếng Việt

## Tổng quan

**CLAP4CLIP** (Continual Learning with Probabilistic Finetuning for Vision-Language Models) là một framework finetuning có tính xác suất cho mô hình CLIP được pretrained, được thiết kế để xử lý các tác vụ **học gia tăng lớp** (class-incremental learning).

Mô hình này đã được chấp nhận tại **NeurIPS 2024** và được xuất bản trong bài báo: [CLAP4CLIP: Continual Learning with Probabilistic Finetuning for Vision-Language Models](https://arxiv.org/pdf/2403.19137v2)

---

## Vấn đề mà CLAP4CLIP giải quyết

Trong **class-incremental learning**, mô hình cần học các lớp mới theo từng task (session) mà không quên các lớp đã học trước đó. Đây là một thách thức lớn vì:

1. **Catastrophic Forgetting**: Mô hình có xu hướng quên kiến thức cũ khi học kiến thức mới
2. **Stability Gap**: Hiệu suất trên các task mới thường không ổn định
3. **Uncertainty Modeling**: Cần mô hình hóa độ không chắc chắn để phát hiện dữ liệu mới và chọn exemplar hiệu quả

---

## Kiến trúc của CLAP4CLIP

### 1. **Nền tảng: CLIP Pretrained**

CLAP4CLIP xây dựng trên mô hình CLIP đã được pretrained:

- **Image Encoder**: Mã hóa ảnh thành vector đặc trưng (image features)
- **Text Encoder**: Mã hóa text prompts thành vector đặc trưng (text features)
- **Contrastive Learning**: So sánh similarity giữa image và text features để phân loại

### 2. **VGA (Visual-Guided Attention)**

VGA là một **Transformer Decoder** có vai trò:

- Nhận đầu vào: text features (query) và image features (key/value)
- Điều chỉnh text features dựa trên ngữ cảnh hình ảnh
- Giúp text features trở nên phù hợp hơn với image features cụ thể

```python
# VGA sử dụng Transformer Decoder
vga_features = self.vga(text_features, image_features)
adjusted_text_features = text_features + vga_features
```

### 3. **Variational Adapters**

Đây là thành phần **core** của CLAP4CLIP, giúp mô hình hóa **uncertainty** (độ không chắc chắn):

#### Cấu trúc:

- **Mu Adapter**: Tạo giá trị trung bình (mean) của phân phối
- **Sigma Adapter**: Tạo độ lệch chuẩn (standard deviation) của phân phối
- Tạo ra một **phân phối chuẩn** (Normal distribution) trên text features

```python
# Tạo phân phối
mu = mu_adapter(text_features)      # Giá trị trung bình
sigma = sigma_adapter(text_features) # Độ lệch chuẩn
dist = Normal(mu, sigma)              # Phân phối chuẩn

# Sampling từ phân phối
samples = dist.rsample([n_samples])  # Lấy mẫu nhiều lần
```

#### Lợi ích:

- **Uncertainty Quantification**: Đo được độ không chắc chắn của dự đoán
- **Ensemble Effect**: Sampling nhiều lần tạo ra hiệu ứng ensemble, cải thiện độ chính xác
- **OOD Detection**: Có thể phát hiện dữ liệu ngoài phân phối (out-of-distribution)

### 4. **Task Tokens (Expandable Tokens)**

Mỗi task có một **task token** riêng:

- Giúp phân biệt các task khác nhau
- Có thể mở rộng khi thêm task mới
- Được sử dụng trong attention mask để ngăn các task tương tác với nhau

### **Lưu ý quan trọng về Task ID:**

CLAP4CLIP **KHÔNG** cần task ID được cung cấp trực tiếp từ bên ngoài. Thay vào đó:

1. **Tự động suy luận Task ID từ Class ID**:

   - Mô hình biết `task_to_cls_num`: mỗi task có bao nhiêu lớp
   - Ví dụ: Task 0 = lớp 0-9, Task 1 = lớp 10-19, Task 2 = lớp 20-29, ...
   - Từ class ID (label), mô hình tự tính được task ID bằng cách xem class ID nằm trong khoảng nào

2. **Class-to-Task Mapping**:

   ```python
   # Tự động tạo mapping: class_id -> task_id
   self.class_to_task_mapping = {
       0: 0, 1: 0, ..., 9: 0,    # Task 0: lớp 0-9
       10: 1, 11: 1, ..., 19: 1, # Task 1: lớp 10-19
       ...
   }
   ```

3. **Trong training và inference**:
   - Chỉ cần **labels** (class IDs), không cần task IDs
   - Mô hình tự động xác định task từ class ID
   - Code: `if ((labels >= lo) & (labels < hi)).any():` - kiểm tra class ID thuộc task nào

### 5. **Hierarchical Modeling (Tùy chọn)**

CLAP4CLIP hỗ trợ kiến trúc phân cấp:

- **Global Level**: Một adapter chung cho tất cả các task
- **Task Level**: Mỗi task có adapter riêng
- Giúp chia sẻ kiến thức giữa các task và tối ưu hóa hiệu suất

### 6. **Anchor Routing (Tùy chọn - Có thể tắt)**

Một cơ chế thông minh để **phân bổ số lần sampling** cho từng task dựa trên độ tương đồng:

#### **Khi BẬT Anchor Routing:**

- Tính độ tương đồng giữa image features với **anchor** của mỗi task
- Phân bổ số samples theo trọng số: task có độ tương đồng cao → nhiều samples hơn
- Tổng số samples vẫn ~ `forward_times` nhưng được phân bổ thông minh
- **Lợi ích**: Tối ưu hóa computation, tập trung vào task phù hợp hơn

#### **Khi TẮT Anchor Routing:**

- Tất cả các task đều dùng `forward_times` cố định (ví dụ: 20 samples cho mỗi task)
- **Đơn giản hơn** nhưng có thể không tối ưu

#### **Tại sao không dùng task_id trực tiếp?**

**Câu hỏi quan trọng**: Tại sao không dùng `task_id` từ labels để phân bổ samples luôn mà lại cần anchor?

**Trả lời**:

1. **Trong test/inference - Không có labels**:

   - Khi test, mô hình **không biết** label/task_id trước
   - Phải dựa vào **visual similarity** để đoán task nào có khả năng cao nhất
   - Anchor routing tính độ tương đồng giữa image features với anchor của mỗi task

2. **Visual similarity quan trọng hơn task_id**:

   - Một image có thể liên quan đến **nhiều task** (ví dụ: ảnh có nhiều object từ các task khác nhau)
   - Task_id chỉ biết task của label, nhưng không biết **mức độ relevance** của các task khác
   - Anchor routing dựa trên **image features** (visual similarity), tự động phát hiện task nào liên quan

3. **Anchor routing thông minh hơn**:

   - **Task_id approach**: Chỉ sample cho task có label → bỏ qua task khác có thể liên quan
   - **Anchor routing**: Dựa trên visual similarity → phân bổ samples cho tất cả task có liên quan
   - Ví dụ: Ảnh "dog" có thể giống task "animals" (task 0) và task "pets" (task 3) → anchor routing sẽ phân bổ samples cho cả 2

4. **Trong code**:
   ```python
   # Anchor routing dựa trên image features, KHÔNG phải labels
   d_weights = self._get_anchor_weights(image_features_normed)  # [B, T]
   # Chỉ dùng labels để đảm bảo task có trong batch có ít nhất 1 sample
   if ((labels >= lo) & (labels < hi)).any():
       if alloc[ti].item() == 0:
           alloc[ti] = 1
   ```

#### **Có thừa không?**

- **KHÔNG thừa**, nhưng là **tùy chọn tối ưu hóa**
- Mô hình vẫn hoạt động bình thường khi tắt (dùng `forward_times` cố định)
- Khi bật có thể cải thiện hiệu quả và hiệu suất, đặc biệt khi có nhiều task
- **Quan trọng nhất**: Hoạt động được trong test khi không có labels
- Có thể tắt bằng flag: `use_anchor_routing=False` (mặc định: `True`)

---

## Quy trình hoạt động

### **Giai đoạn Training:**

1. **Encode Image**:

   - Ảnh được mã hóa qua Image Encoder thành image features

2. **Encode Text**:

   - Text prompts được mã hóa qua Text Encoder thành text features
   - Text features được điều chỉnh bởi VGA dựa trên image features

3. **Variational Sampling**:

   - Text features được đưa qua Mu và Sigma Adapters
   - Tạo phân phối chuẩn và lấy mẫu nhiều lần (ví dụ: 20 lần)
   - Mỗi sample được cộng vào text features gốc

4. **Tính Logits**:

   - Với mỗi sample, tính similarity giữa image features và text features
   - Lấy trung bình của tất cả các logits từ các samples

5. **Loss Function**:
   - **Cross-Entropy Loss**: Loss phân loại chính
   - **KL Divergence Loss**: Khuyến khích phân phối gần với prior distribution
   - **Prior Matching Loss**: Giúp phân phối không quá xa prior

### **Giai đoạn Inference:**

1. Tương tự như training nhưng:
   - Không cần gradient
   - Có thể sử dụng anchor routing để phân bổ samples hiệu quả
   - Tính trung bình các logits từ tất cả các samples

---

## Các tính năng đặc biệt

### 1. **Class-Incremental Learning**

- Mỗi task thêm các lớp mới
- Mô hình học task mới mà không quên task cũ
- Sử dụng **exemplar selection** để lưu trữ dữ liệu đại diện từ các task cũ

### 2. **Uncertainty-Aware Exemplar Selection**

CLAP4CLIP sử dụng uncertainty để chọn exemplar tốt hơn:

- **Energy Score**: Dựa trên log-sum-exp của logits
- **Variance**: Độ biến thiên của các samples
- **Entropy**: Entropy của phân phối dự đoán

### 3. **Out-of-Distribution Detection**

Mô hình có thể phát hiện dữ liệu mới (novel data) bằng cách:

- So sánh confidence score giữa ID (in-distribution) và OOD data
- Sử dụng Energy, AUROC, AUPR để đánh giá

### 4. **Knowledge Distillation**

Khi học task mới, mô hình có thể:

- Lưu lại adapter của task cũ
- Sử dụng distillation loss để giữ kiến thức cũ
- Tránh catastrophic forgetting

---

## Các thành phần chính trong code

### **1. CLIP Class** (`classifier/continual_clip_variational.py`)

```python
class CLIP(nn.Module):
    def __init__(self, args, class_names, clip_model, vga,
                 mu_adapters, sigma_adapters, task_tokens, ...):
        # Image encoder (frozen)
        self.image_encoder = clip_model.visual

        # VGA: Transformer Decoder
        self.vga = vga

        # Variational Adapters
        self.mu_adapters = mu_adapters      # Mu adapters cho mỗi task
        self.sigma_adapters = sigma_adapters # Sigma adapters cho mỗi task

        # Task tokens
        self.task_tokens = task_tokens
```

### **2. Forward Pass**

```python
def forward(self, image, labels, test=False):
    # 1. Encode image
    image_features = self.image_encoder(image)

    # 2. Get text features
    text_features = self.frozen_text_features

    # 3. VGA adjustment
    vga_features = self.vga(text_features, image_features)
    text_features = text_features + vga_features

    # 4. Variational sampling
    for task_id in range(num_tasks):
        qdist = self.get_variational_adapter_features(text_features, task_id)
        samples = qdist.rsample([n_samples])  # Lấy mẫu
        text_features_sampled = text_features + samples

        # 5. Tính logits
        logits = image_features @ text_features_sampled.t()

    return logits.mean(0)  # Trung bình các samples
```

### **3. Adapter Structure**

```python
class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim, sigma=False):
        self.fc = nn.Linear(in_dim, out_dim)
        self.sigma = sigma  # Nếu True, dùng softplus để đảm bảo sigma > 0

    def forward(self, x):
        if self.sigma:
            return F.softplus(self.fc(x)) * 0.999 + 0.001
        else:
            return self.fc(x)
```

---

## Ưu điểm của CLAP4CLIP

1. **Tính tổng quát**: Hỗ trợ nhiều loại prompt style (hand-crafted, CoOp, MaPLe, AttriCLIP)
2. **Uncertainty Modeling**: Có thể đo và sử dụng uncertainty
3. **Hiệu suất cao**: Cải thiện đáng kể so với các phương pháp baseline
4. **Linh hoạt**: Có thể mở rộng adapter và task tokens theo từng task
5. **Out-of-the-box**: Có thể sử dụng ngay cho OOD detection và exemplar selection

---

## Kết luận

CLAP4CLIP là một framework mạnh mẽ cho class-incremental learning với CLIP, kết hợp:

- **Probabilistic modeling** (variational adapters)
- **Visual-guided attention** (VGA)
- **Task-specific components** (task tokens, expandable adapters)
- **Uncertainty-aware mechanisms** (sampling, OOD detection)

Framework này không chỉ giải quyết vấn đề catastrophic forgetting mà còn cung cấp các công cụ để làm việc với uncertainty và phát hiện dữ liệu mới.
