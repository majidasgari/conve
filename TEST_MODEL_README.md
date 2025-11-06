# راهنمای تست مدل ConvE

این اسکریپت‌ها برای لود کردن و تست مدل‌های آموزش‌دیده ConvE طراحی شده‌اند.

## فایل‌ها

- `test_model.py`: اسکریپت اصلی Python برای لود و تست مدل
- `test_model.sh`: اسکریپت bash برای اجرای سریع با پارامترهای پیش‌فرض

## نحوه استفاده

### روش 1: استفاده از اسکریپت bash (راحت‌تر)

```bash
./test_model.sh
```

### روش 2: اجرای مستقیم Python

```bash
CUDA_VISIBLE_DEVICES=0 python test_model.py \
    --model conve \
    --data FB15k-237 \
    --model-path saved_models/FB15k-237_conve_0.2_0.3.model \
    --input-drop 0.2 \
    --hidden-drop 0.3 \
    --feat-drop 0.2 \
    --cuda
```

### روش 3: تست با پارامترهای سفارشی

```bash
python test_model.py --help
```

این دستور تمام پارامترهای قابل تنظیم را نمایش می‌دهد.

## پارامترهای مهم

| پارامتر | توضیح | مقدار پیش‌فرض |
|---------|-------|---------------|
| `--model` | نوع مدل (conve, distmult, complex) | conve |
| `--data` | نام dataset | FB15k-237 |
| `--model-path` | مسیر فایل مدل ذخیره‌شده | saved_models/FB15k-237_conve_0.2_0.3.model |
| `--input-drop` | Dropout برای input embeddings | 0.2 |
| `--hidden-drop` | Dropout برای hidden layer | 0.3 |
| `--feat-drop` | Dropout برای convolutional features | 0.2 |
| `--embedding-dim` | بعد embedding | 200 |
| `--test-batch-size` | اندازه batch برای تست | 128 |
| `--cuda` | استفاده از GPU | فعال |

## خروجی

اسکریپت خروجی‌های زیر را نمایش می‌دهد:

1. **اطلاعات مدل**: تعداد entities، relations، و پارامترها
2. **نتایج Test Set**: 
   - Hits@1, Hits@3, Hits@10
   - Mean Rank (MR)
   - Mean Reciprocal Rank (MRR)
3. **نتایج Dev Set**: معیارهای مشابه

همچنین نتایج در فایل log مربوط به evaluation ذخیره می‌شوند.

## مثال خروجی

```
======================================================================
شروع تست مدل
======================================================================

تعداد Entities: 14541
تعداد Relations: 237

ایجاد مدل: conve
بارگذاری مدل از: saved_models/FB15k-237_conve_0.2_0.3.model
مدل به GPU منتقل شد

اطلاعات مدل:
تعداد کل پارامترها: 4,932,541

======================================================================
ارزیابی روی Test Set
======================================================================
...
Hits @10: 0.49
Mean rank: 246
Mean reciprocal rank: 0.32
...
```

## نکات مهم

1. **پارامترهای معماری باید دقیقاً با مدل آموزش‌داده شده یکسان باشند**
   - اگر مدل با embedding-dim متفاوت آموزش دیده، باید همان مقدار را مشخص کنید

2. **فایل مدل باید وجود داشته باشد**
   - مسیر پیش‌فرض: `saved_models/FB15k-237_conve_0.2_0.3.model`
   - برای مدل‌های دیگر، مسیر را با `--model-path` مشخص کنید

3. **Dataset باید از قبل preprocess شده باشد**
   - اگر اولین بار است، ابتدا دستور زیر را اجرا کنید:
   ```bash
   python main.py --data FB15k-237 --preprocess
   ```

## رفع مشکلات رایج

### خطا: فایل مدل یافت نشد
```bash
# بررسی مدل‌های موجود
ls -lh saved_models/
```

### خطا: Dataset پیدا نشد
```bash
# اجرای preprocessing
python main.py --data FB15k-237 --preprocess
```

### خطا: CUDA out of memory
```bash
# کاهش batch size
python test_model.py --test-batch-size 64
```

### خطا: Shape mismatch
این خطا نشان می‌دهد پارامترهای معماری اشتباه است. مطمئن شوید که تمام پارامترها (embedding-dim، hidden-size، و غیره) با مدل آموزش‌داده شده یکسان هستند.

## تست مدل‌های دیگر

برای تست مدل‌هایی که با پارامترهای متفاوت آموزش دیده‌اند:

```bash
# مثال: مدل با embedding size 100
python test_model.py \
    --model-path saved_models/MY_MODEL.model \
    --embedding-dim 100 \
    --embedding-shape1 10 \
    --hidden-size 4608
```

## ارتباط با ما

در صورت بروز مشکل، لطفاً:
1. پارامترهای آموزش مدل را بررسی کنید
2. مطمئن شوید که dataset به درستی preprocess شده است
3. فایل log مربوط به evaluation را بررسی کنید
