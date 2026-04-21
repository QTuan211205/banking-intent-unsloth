# 🏦 Banking Intent Classification with Llama-3-8B (Unsloth)

[cite_start]Dự án này chứa mã nguồn cho **Đồ án 2: Tinh chỉnh mô hình phát hiện ý định**, thuộc môn học *Ứng dụng NLP trong công nghiệp*[cite: 8, 10].

* **Sinh viên:** Đào Quốc Tuấn (23120392)
* **Mô hình:** Llama-3-8B (4-bit)
* **Phương pháp:** SFT with LoRA

---

## 📺 Video Demonstration
[cite_start]Video trình diễn thực thi script, ví dụ đầu vào và độ chính xác[cite: 95, 96].

[cite_start]👉 **[Xem Video Demo tại đây](https://drive.google.com/drive/folders/1KO1XpgEF305-Z_Juq8tBXhggoz6UjnIK?hl=vi)** [cite: 103]

---

## 📁 Repository Structure
```text
banking-intent-unsloth/
├── scripts/
[cite_start]│   ├── preprocess_data.py   # Tải và tiền xử lý dữ liệu [cite: 78]
[cite_start]│   ├── train.py             # Huấn luyện mô hình [cite: 76]
[cite_start]│   └── inference.py         # Lớp suy luận độc lập [cite: 77]
├── configs/
[cite_start]│   ├── train.yaml           # Cấu hình huấn luyện [cite: 80]
[cite_start]│   └── inference.yaml       # Cấu hình đường dẫn model [cite: 81]
├── sample_data/
[cite_start]│   ├── train.csv            # 5000 mẫu train [cite: 83]
[cite_start]│   └── test.csv             # 1000 mẫu test [cite: 84]
[cite_start]├── train.sh                 # Bash chạy huấn luyện [cite: 85]
[cite_start]├── inference.sh             # Bash chạy suy luận [cite: 86]
[cite_start]├── requirements.txt         # Thư viện phụ thuộc [cite: 87]
[cite_start]└── README.md                # Tài liệu hướng dẫn [cite: 88]
```

---

## 🛠️ 1. Cài đặt môi trường
[cite_start]Cài đặt các thư viện cần thiết[cite: 89]:

```bash
git clone https://github.com/QTuan211205/banking-intent-unsloth.git
cd banking-intent-unsloth
pip install -r requirements.txt
```

## 📥 2. Chuẩn bị dữ liệu
[cite_start]Tải và lấy mẫu dữ liệu từ BANKING77[cite: 26, 28]:

```bash
python scripts/preprocess_data.py
```

## 🧠 3. Huấn luyện mô hình
[cite_start]Chạy script để tinh chỉnh mô hình với Unsloth[cite: 42]:

```bash
bash train.sh
```

## 📦 4. Tải trọng số (Pre-trained Weights)
Nếu không muốn train lại, hãy chạy lệnh này trên Kaggle để tải model từ Drive:

```python
!pip install gdown
!gdown --folder 1RSHsvAPPlxq6fiXD8pal3JaX1DlDH9Pc -O /kaggle/working/model_weights
```

## 🎯 5. Chạy suy luận (Inference)
[cite_start]Dự đoán ý định từ tin nhắn khách hàng[cite: 53, 54]:

```bash
bash inference.sh
```

[cite_start]**Ví dụ sử dụng (Python):** [cite: 70]
```python
from scripts.inference import IntentClassification

classifier = IntentClassification(model_path="configs/inference.yaml")
message = "I lost my credit card, what should I do?"
print(f"🤖 Dự đoán: {classifier(message)}")
```

---
[cite_start]**🎯 Kết quả:** Độ chính xác tập Test đạt **91.40%**[cite: 100].