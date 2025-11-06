# Phân tích các vấn đề gây accuracy thấp

## 1. Vấn đề về trọng số Loss Function (QUAN TRỌNG NHẤT)

### Vấn đề:

- **Dòng 727**: `ortho_loss` có weight = **5**, quá cao so với cross-entropy loss
- **Dòng 536, 700**: `prior_matching_loss` có weight = **0.001**, có thể quá nhỏ
- **Dòng 1033**: Loss được tính: `loss = F.cross_entropy(output, targets) + kl_loss + prior_matching_loss`
  - Không có weight balancing giữa các loss components
  - `ortho_loss` (weight=5) có thể dominate và làm model không học được classification

### Giải pháp đề xuất:

```python
# Thay đổi weight của ortho_loss từ 5 xuống 0.1-0.5
kl_losses.append(F.cross_entropy(sims, ...) * 0.1)  # thay vì * 5

# Hoặc thêm weight balancing trong loss calculation:
loss = F.cross_entropy(output, targets) + 0.1 * kl_loss + 0.1 * prior_matching_loss
```

## 2. Model ở eval mode ngay sau init

### Vấn đề:

- **Dòng 1237**: `self.model.eval()` được gọi ngay sau khi khởi tạo model
- Mặc dù sau đó có `self.model.train()` ở dòng 971, nhưng có thể có vấn đề với batch normalization/dropout

### Giải pháp:

- Đảm bảo model được set về train mode trước khi training
- Kiểm tra xem có layer nào bị freeze không đúng không

## 3. Learning Rate có thể không phù hợp

### Vấn đề:

- **Dòng 814**: `self.lr = args.lr*args.train_batch/20`
- Nếu `train_batch` nhỏ, learning rate sẽ rất nhỏ
- Nếu `train_batch` lớn, learning rate sẽ lớn

### Giải pháp:

- Kiểm tra giá trị learning rate thực tế
- Có thể cần điều chỉnh công thức tính lr

## 4. Xử lý NaN/Inf che giấu vấn đề

### Vấn đề:

- Code có nhiều chỗ replace NaN/Inf bằng 0 (dòng 168, 421, 642)
- Điều này che giấu vấn đề thay vì fix root cause
- Có thể dẫn đến gradient = 0 và model không học được

### Giải pháp:

- Tìm nguyên nhân gốc gây ra NaN/Inf
- Có thể do:
  - Gradient explosion
  - Division by zero
  - Numerical instability trong KL divergence

## 5. Anchor Routing có thể phân bổ samples sai

### Vấn đề:

- **Dòng 387, 437**: Số samples cho mỗi task được phân bổ dựa trên anchor weights
- Nếu anchor weights không chính xác, một số task có thể không được sample đủ
- **Dòng 402-404**: Có fallback nhưng có thể không đủ

## 6. Shape handling phức tạp

### Vấn đề:

- Code có nhiều xử lý shape phức tạp (dòng 997-1025, 1113-1141)
- Có thể dẫn đến logits không đúng shape
- Nếu shape sai, accuracy sẽ thấp

## Khuyến nghị ưu tiên:

1. **GIẢM WEIGHT CỦA ORTHO_LOSS** (từ 5 xuống 0.1-0.5) - Đây là vấn đề quan trọng nhất
2. **THÊM WEIGHT BALANCING** cho loss function
3. **KIỂM TRA LEARNING RATE** thực tế trong quá trình training
4. **LOG CÁC LOSS COMPONENTS** để xem loss nào dominate
5. **KIỂM TRA SHAPE CỦA LOGITS** trong quá trình training
