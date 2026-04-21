#!/bin/bash

echo "========================================="
echo "🚀 BẮT ĐẦU CHUẨN BỊ DỮ LIỆU"
echo "========================================="
python scripts/preprocess_data.py

echo ""
echo "========================================="
echo "🧠 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH (FINE-TUNING)"
echo "========================================="
python scripts/train.py

echo ""
echo "🎉 Quá trình huấn luyện đã hoàn tất!"