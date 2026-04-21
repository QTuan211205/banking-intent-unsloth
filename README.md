# 🏦 Banking Intent Classification with Llama-3-8B (Unsloth)

Dự án này chứa mã nguồn cho **Đồ án 2: Tinh chỉnh mô hình phát hiện ý định**, thuộc môn học *Ứng dụng NLP trong công nghiệp*.

* **Sinh viên:** Đào Quốc Tuấn (23120392)
* **Mô hình:** Llama-3-8B (4-bit)
* **Phương pháp:** SFT with LoRA

---

## 📺 Video Demonstration
Video trình diễn thực thi script, ví dụ đầu vào và độ chính xác.

👉 [**Xem Video Demo tại đây**](https://drive.google.com/drive/folders/1KO1XpgEF305-Z_Juq8tBXhggoz6UjnIK?hl=vi)

---

## 📁 Repository Structure
```text
banking-intent-unsloth/
├── scripts/
│   ├── preprocess_data.py   # Tải và tiền xử lý dữ liệu
│   ├── train.py             # Huấn luyện mô hình
│   └── inference.py         # Lớp suy luận độc lập
├── configs/
│   ├── train.yaml           # Cấu hình huấn luyện
│   └── inference.yaml       # Cấu hình đường dẫn model
├── sample_data/
│   ├── train.csv            # 5000 mẫu train
│   └── test.csv             # 1000 mẫu test
├── train.sh                 # Bash chạy huấn luyện
├── inference.sh             # Bash chạy suy luận
├── requirements.txt         # Thư viện phụ thuộc
└── README.md                # Tài liệu hướng dẫn
🛠️ 1. Cài đặt môi trường
Cài đặt các thư viện cần thiết trên Kaggle/Colab:

Bash
git clone [https://github.com/QTuan211205/banking-intent-unsloth.git](https://github.com/QTuan211205/banking-intent-unsloth.git)
cd banking-intent-unsloth
pip install -r requirements.txt
📦 2. Tải trọng số (Pre-trained Weights)
Nếu không muốn train lại, chạy lệnh này trong Notebook để tải model từ Drive:

Python
!pip install gdown
!gdown --folder 1RSHsvAPPlxq6fiXD8pal3JaX1DlDH9Pc -O /kaggle/working/model_weights
🎯 3. Chạy suy luận (Inference)
Thực thi script để dự đoán ý định:

Bash
bash inference.sh
Ví dụ sử dụng (Python):

Python
from scripts.inference import IntentClassification

classifier = IntentClassification(model_path="configs/inference.yaml")
message = "I lost my credit card, what should I do?"
print(f"🤖 Dự đoán: {classifier(message)}")
🎯 Kết quả: Độ chính xác tập Test đạt 91.40%.