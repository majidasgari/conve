#!/bin/bash
# اسکریپت برای تست مدل ConvE

# فعال‌سازی virtual environment اگر وجود دارد
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# اجرای تست با پارامترهای مشابه آموزش
# توجه: مدل بدون --use-bias آموزش دیده است
CUDA_VISIBLE_DEVICES=0 python test_model.py \
    --model conve \
    --data FB15k-237 \
    --model-path saved_models/FB15k-237_conve_0.2_0.3.model \
    --input-drop 0.2 \
    --hidden-drop 0.3 \
    --feat-drop 0.2 \
    --embedding-dim 200 \
    --embedding-shape1 20 \
    --hidden-size 9728 \
    --test-batch-size 128 \
    --cuda
