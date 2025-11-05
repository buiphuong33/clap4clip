# Phân tích tại sao accuracy giảm khi thêm Anchor Routing

## 🔴 Lỗi nghiêm trọng: THÊM `.mean(0)` không có trong code gốc

### **So sánh code gốc vs code có anchor:**

#### **1. Test Mode:**

**Code gốc:**

```python
rsamples = qdist.rsample([self.forward_times])  # [forward_times, n_classes, D]
text_features_ = text_features_.unsqueeze(0).expand(self.forward_times, -1, -1)  # [forward_times, n_classes, D]
text_features_ = rsamples + text_features_  # [forward_times, n_classes, D]
logits_ = logit_scale * image_features_normed @ text_features_.permute(0, 2, 1)  # [B, forward_times, n_classes]
logits.append(logits_)  # ❌ KHÔNG có .mean(0)
```

**Code có anchor (SAI):**

```python
n_samples_task = int(alloc[i].item())  # Có thể khác forward_times
rsamples = qdist.rsample([n_samples_task])  # [n_samples_task, n_classes, D]
text_features_ = text_features_.unsqueeze(0).expand(n_samples_task, -1, -1)  # [n_samples_task, n_classes, D]
text_features_ = rsamples + text_features_  # [n_samples_task, n_classes, D]
logits_ = logit_scale * image_features_normed @ text_features_.permute(0, 2, 1)  # [B, n_samples_task, n_classes]
logits_ = logits_.mean(0, keepdim=True)  # ❌ THÊM DÒNG NÀY KHÔNG CÓ TRONG CODE GỐC!
logits.append(logits_)
```

#### **2. Training Mode:**

**Code gốc:**

```python
rsamples = qdist.rsample([self.forward_times])  # [forward_times, n_classes, D]
text_features_ = text_features_.unsqueeze(0).expand(self.forward_times, -1, -1)  # [forward_times, n_classes, D]
text_features_ = rsamples + text_features_  # [forward_times, n_classes, D]
logits_ = (logit_scale * image_features_normed @ text_features_.permute(0, 2, 1))  # [B, forward_times, n_classes]
logits.append(logits_)  # ❌ KHÔNG có .mean(0)
```

**Code có anchor (SAI):**

```python
n_samples_task = int(alloc[i].item())  # Có thể khác forward_times
rsamples = qdist.rsample([n_samples_task])  # [n_samples_task, n_classes, D]
text_features_ = text_features_.unsqueeze(0).expand(n_samples_task, -1, -1)  # [n_samples_task, n_classes, D]
text_features_ = rsamples + text_features_  # [n_samples_task, n_classes, D]
logits_ = (logit_scale * image_features_normed @ text_features_.permute(0, 2, 1))  # [B, n_samples_task, n_classes]
logits_ = logits_.mean(0, keepdim=True)  # ❌ THÊM DÒNG NÀY KHÔNG CÓ TRONG CODE GỐC!
logits.append(logits_)
```

---

## 🔴 Vấn đề chính:

### **1. Code gốc KHÔNG có `.mean(0)` trước khi append:**

Trong code gốc, `logits_` có shape `[B, forward_times, n_classes]` và được append trực tiếp vào list `logits`.

Sau đó, khi `torch.cat(logits, -1)`, nó sẽ concat theo chiều class, tạo ra shape `[B, forward_times, total_classes]`.

Cuối cùng, trong code gốc có thể có `.mean(0)` ở đâu đó, hoặc có thể không.

**Nhưng quan trọng:** Code gốc giữ nguyên shape `[B, forward_times, n_classes]` cho mỗi task.

### **2. Code có anchor THÊM `.mean(0)` sớm:**

Khi thêm `.mean(0, keepdim=True)`, shape của `logits_` trở thành `[1, n_classes]` (hoặc `[B, n_classes]` nếu không có keepdim).

**Vấn đề:**

- Code gốc: `logits_` shape `[B, forward_times, n_classes]` → sau khi cat: `[B, forward_times, total_classes]`
- Code có anchor: `logits_` shape `[1, n_classes]` (hoặc `[B, n_classes]`) → sau khi cat: `[1, total_classes]` hoặc `[B, total_classes]`

**Điều này làm thay đổi hoàn toàn cách tính toán!**

### **3. Vấn đề về số lượng samples khác nhau:**

Khi dùng anchor routing, mỗi task có số samples khác nhau (`n_samples_task`). Điều này có nghĩa:

- Task 0: 15 samples
- Task 1: 3 samples
- Task 2: 2 samples

Khi tính `.mean(0)` trên các samples khác nhau, trọng số của mỗi task sẽ khác nhau, dẫn đến kết quả không công bằng.

---

## ✅ Giải pháp:

### **Option 1: Loại bỏ `.mean(0)` để giống code gốc**

```python
# Test mode (dòng 427-433)
logits_ = logit_scale * image_features_normed @ text_features_.permute(0, 2, 1)
# XÓA dòng này: logits_ = logits_.mean(0, keepdim=True)
logits.append(logits_)

# Training mode (dòng 577-596)
logits_ = (logit_scale * image_features_normed @ text_features_.permute(0, 2, 1))
# XÓA dòng này: logits_ = logits_.mean(0, keepdim=True)
logits.append(logits_)
```

Sau đó, ở cuối forward function, tính mean nếu cần:

```python
logits = torch.cat(logits, -1)  # [B, n_samples_varies, total_classes]
logits = logits.mean(1)  # Mean over samples dimension
```

### **Option 2: Giữ `.mean(0)` nhưng đảm bảo tất cả task có cùng số samples**

Nếu muốn giữ `.mean(0)`, phải đảm bảo tất cả task có cùng số samples, hoặc normalize theo số samples.

### **Option 3: Điều chỉnh allocation để tổng samples không đổi**

Thay vì phân bổ samples theo weights, giữ tổng số samples = `forward_times * num_tasks` và phân bổ đều hơn.

---

## 🔍 Nguyên nhân chính xác:

**Accuracy giảm vì:**

1. **Thêm `.mean(0)` sớm** → Thay đổi cách tính toán, mất thông tin về variance giữa các samples
2. **Số samples khác nhau** → Mỗi task có trọng số khác nhau khi tính mean, không công bằng
3. **Shape không đúng** → Khi cat logits, shape không match với code gốc

**Giải pháp đơn giản nhất:** Xóa `.mean(0, keepdim=True)` ở cả test và training mode để giống code gốc.
