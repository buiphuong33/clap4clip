# Lỗi trong Anchor Routing Implementation

## 🔴 Lỗi nghiêm trọng 1: Khi task bị skip, logits không được thêm vào

**Vị trí**: Dòng 414-415 trong `continual_clip_variational.py`

```python
if n_samples_task == 0:
    continue  # ❌ Task bị skip hoàn toàn!
```

**Vấn đề**:

- Khi một task có `alloc[i] = 0`, task đó bị skip hoàn toàn
- `logits.append(logits_)` không được gọi cho task đó
- Khi `logits = torch.cat(logits, -1)` ở dòng 438, số lượng logits sẽ ít hơn số lượng task
- **Shape không đúng**: Ví dụ nếu có 3 task nhưng task 1 bị skip, logits sẽ chỉ có 2 phần, dẫn đến lỗi shape mismatch

**Fix**:

```python
# Thay vì skip, phải thêm logits với giá trị zero hoặc -inf
if n_samples_task == 0:
    # Tạo logits zero cho task này để giữ shape
    num_classes_task = self.task_to_cls_num[i]
    logits_ = torch.zeros((image_features_normed.shape[0], num_classes_task),
                          device=image_features_normed.device, dtype=image_features_normed.dtype)
    logits_.fill_(-float('inf'))  # Hoặc dùng giá trị rất nhỏ
    logits.append(logits_)
    continue
```

---

## 🔴 Lỗi nghiêm trọng 2: taskwise_means thiếu khi task bị skip

**Vị trí**: Dòng 558-574 trong training mode

**Vấn đề**:

- Khi task bị skip, `taskwise_means.append(rsamples.mean(0))` không được gọi
- Ở dòng 606, `taskwise_means = torch.cat(taskwise_means)` sẽ lỗi nếu số lượng taskwise_means không đúng

**Fix**:

```python
if n_samples_task == 0:
    # Phải append một giá trị giả cho taskwise_means
    taskwise_means.append(torch.zeros_like(text_features_relevant.mean(0)))
    # ... phần logits như trên
    continue
```

---

## 🟡 Lỗi tiềm ẩn 3: Allocation không chính xác

**Vị trí**: Dòng 378, 511

```python
alloc = (avg_w * self.forward_times).round().to(torch.int64)
```

**Vấn đề**:

- Tổng của `alloc` có thể không bằng `forward_times`
- Có thể nhỏ hơn (nếu nhiều task có weight nhỏ) hoặc lớn hơn (nếu rounding)
- Dẫn đến số lượng samples không đúng như mong đợi

**Fix**:

```python
# Đảm bảo tổng bằng forward_times
alloc = (avg_w * self.forward_times).round().to(torch.int64)
total = alloc.sum().item()
if total != self.forward_times:
    # Điều chỉnh task có weight cao nhất
    diff = self.forward_times - total
    top_idx = torch.argmax(avg_w).item()
    alloc[top_idx] += diff
```

---

## 🟡 Lỗi tiềm ẩn 4: Image anchor chưa được khởi tạo trong test

**Vị trí**: Dòng 345, 269-294

**Vấn đề**:

- Image anchor chỉ được update khi có labels (dòng 345: `if labels is not None`)
- Trong test mode, có thể không có labels
- Khi gọi `_get_anchor_weights`, image anchor có thể chưa có → dùng fallback (mean của batch), không chính xác

**Fix**:

```python
# Trong test mode, cũng cần update anchor nếu có thể
if self.use_anchor_routing:
    if labels is not None:
        self._update_image_anchors(image_features_normed, labels)
    # Hoặc update dựa trên prediction nếu không có labels
```

---

## 🟡 Lỗi tiềm ẩn 5: compute_ram thiếu khi task bị skip

**Vị trí**: Dòng 436, 600

**Vấn đề**:

- Khi task bị skip, `samplewise_text_feats.append(text_features_relevant)` không được gọi
- Có thể dẫn đến lỗi shape khi compute RAM

**Fix**: Tương tự như logits, phải append giá trị giả

---

## ✅ Giải pháp tổng thể

**Thay vì skip task hoàn toàn, nên đảm bảo mỗi task có ít nhất 1 sample:**

```python
# Đảm bảo mỗi task có ít nhất 1 sample
if alloc.sum().item() == 0:
    top = torch.argmax(avg_w).item()
    alloc[top] = 1
else:
    # Đảm bảo task có trong batch có ít nhất 1 sample
    for ti, (lo, hi) in enumerate(bounds):
        if ((labels >= lo) & (labels < hi)).any():
            if alloc[ti].item() == 0:
                alloc[ti] = 1

    # Điều chỉnh để tổng vẫn ~ forward_times
    total = alloc.sum().item()
    if total > self.forward_times:
        # Giảm samples của task có weight thấp nhất
        while total > self.forward_times and alloc.min().item() > 1:
            min_idx = torch.argmin(alloc).item()
            alloc[min_idx] -= 1
            total -= 1
    elif total < self.forward_times:
        # Tăng samples của task có weight cao nhất
        max_idx = torch.argmax(alloc).item()
        alloc[max_idx] += (self.forward_times - total)
```

**Hoặc đơn giản hơn: Đảm bảo không bao giờ có n_samples_task == 0:**

```python
n_samples_task = int(alloc[i].item()) if (self.use_anchor_routing and alloc is not None) else self.forward_times
n_samples_task = max(1, n_samples_task)  # ✅ Đảm bảo ít nhất 1 sample
```

---

## 📝 Tóm tắt

1. **Lỗi nghiêm trọng nhất**: Khi task bị skip, logits không được thêm → shape mismatch
2. **Lỗi nghiêm trọng thứ 2**: taskwise_means thiếu → lỗi khi cat
3. **Lỗi tiềm ẩn**: Allocation không chính xác, image anchor chưa khởi tạo

**Giải pháp đơn giản nhất**: Đảm bảo `n_samples_task >= 1` cho mọi task, không bao giờ skip task hoàn toàn.
