# مقایسه دیتاست‌های FB15k-237 و FarsPredict

## تعداد داده‌ها

### FB15k-237:
- Train: 272,115 triples
- Valid: 17,535 triples  
- Test: 20,466 triples
- **مجموع**: 310,116 triples
- Entities: 14,544
- Relations: 477

### FarsPredict:
- Train: 435,600 triples (1.60x بیشتر از FB15k-237)
- Valid: 62,228 triples (3.55x بیشتر)
- Test: 124,459 triples (6.08x بیشتر)
- **مجموع**: 622,287 triples (2.01x بیشتر)
- Entities: 107,827 (7.41x بیشتر)
- Relations: 392 (0.82x - کمتر)

## حجم فایل‌ها

### FB15k-237:
- فایل‌های JSON preprocessing: 238 MB
- مجموع کل: 264 MB

### FarsPredict:
- فایل‌های JSON preprocessing: 21 GB (88x بیشتر!)
  - e1rel_to_e2_ranking_test.json: 14 GB (145x بیشتر از FB15k-237)
  - e1rel_to_e2_ranking_dev.json: 6.7 GB (81x بیشتر)
  - e1rel_to_e2_full.json: 168 MB (5.4x بیشتر)
  - e1rel_to_e2_train.json: 124 MB (4.4x بیشتر)
- مجموع کل: 21 GB

## دلایل کندی Training

### 1. **تعداد Entities بسیار بیشتر** (7.41x):
   - 14,544 → 107,827 entities
   - لایه آخر مدل (embedding matrix) باید 107,827 entity را نگه دارد
   - حجم محاسبات در هر batch بسیار بیشتر می‌شود

### 2. **حجم فایل‌های Ranking** (81-145x بزرگتر):
   - این فایل‌ها برای evaluation استفاده می‌شوند
   - بارگذاری و پردازش آن‌ها زمان بیشتری می‌برد

### 3. **داده‌های Training بیشتر** (1.60x):
   - 435,600 vs 272,115 triples
   - هر epoch زمان بیشتری می‌برد

### 4. **Memory Usage**:
   - مدل FarsPredict: 91 MB
   - مدل FB15k-237: 19 MB
   - **4.8x بزرگتر** به دلیل embedding matrix بزرگتر

## پیش‌بینی زمان Training

اگر FB15k-237 X ساعت طول کشید:
- با توجه به داده‌های بیشتر: ~1.6X
- با توجه به entities بیشتر: ~2-3X (محاسبات پیچیده‌تر)
- با توجه به evaluation سنگین‌تر: ~2-3X

**تخمین کلی**: **3-5 برابر** زمان FB15k-237

اگر FB15k-237 مثلاً 5 ساعت طول کشیده، FarsPredict می‌تواند **15-25 ساعت** طول بکشد.

## توصیه‌ها برای تسریع

1. **افزایش batch size** (اگر GPU memory اجازه دهد)
2. **کاهش تعداد epochs**
3. **استفاده از evaluation کمتر** در حین training
4. **استفاده از GPU قوی‌تر**
