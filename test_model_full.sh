#!/bin/bash
# اسکریپت کامل برای preprocess و تست مدل

# فعال‌سازی virtual environment
source .venv/bin/activate

echo "======================================================================"
echo "بررسی وضعیت Dataset"
echo "======================================================================"

# بررسی اینکه آیا dataset preprocess شده است
DATA_DIR="/home/ubuntu/.data/FB15k-237"

if [ ! -d "$DATA_DIR" ] || [ ! -f "$DATA_DIR/vocab/e1.pkl" ]; then
    echo "⚠️  Dataset preprocess نشده است. شروع preprocessing..."
    echo ""
    CUDA_VISIBLE_DEVICES=0 python main.py --data FB15k-237 --preprocess
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ خطا در preprocessing!"
        exit 1
    fi
    echo ""
    echo "✓ Preprocessing با موفقیت انجام شد"
else
    echo "✓ Dataset از قبل preprocess شده است"
fi

echo ""
echo "======================================================================"
echo "شروع تست مدل"
echo "======================================================================"
echo ""

# اجرای تست
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
    --use-bias \
    --test-batch-size 128 \
    --cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ تست با موفقیت انجام شد!"
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "❌ خطا در اجرای تست!"
    echo "======================================================================"
    exit 1
fi
