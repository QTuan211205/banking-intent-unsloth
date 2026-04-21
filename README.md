🏦 Banking Intent Classification with Llama-3-8B (Unsloth)

Dự án này chứa mã nguồn cho Đồ án 2: Tinh chỉnh mô hình phát hiện ý định với tập dữ liệu ngân hàng, thuộc môn học Ứng dụng Xử lý ngôn ngữ tự nhiên trong công nghiệp.

Mục tiêu của dự án là tinh chỉnh Mô hình ngôn ngữ lớn (Llama-3-8B) bằng Unsloth và QLoRA để phân loại chính xác các truy vấn của khách hàng vào các ý định ngân hàng cụ thể bằng tập dữ liệu BANKING77.

Sinh viên thực hiện: Đào Quốc Tuấn (23120392)

Mô hình: unsloth/llama-3-8b-bnb-4bit (Lượng tử hóa 4-bit)

Phương pháp: Tinh chỉnh có giám sát (SFT) với LoRA

📺 Video Demonstration

Video dưới đây trình bày việc thực thi script suy luận, hiển thị các ví dụ đầu vào, nhãn ý định dự đoán và độ chính xác cuối cùng trên tập kiểm thử.

👉 Xem Video Demo tại đây

📁 Repository Structure

banking-intent-unsloth/
├── scripts/
│   ├── preprocess_data.py   # Script tải, làm sạch và lấy mẫu tập dữ liệu
│   ├── train.py             # Script cấu hình và chạy SFTTrainer
│   └── inference.py         # Triển khai lớp suy luận độc lập
├── configs/
│   ├── train.yaml           # Siêu tham số và cấu hình huấn luyện
│   └── inference.yaml       # Cấu hình suy luận (đường dẫn mô hình)
├── sample_data/
│   ├── train.csv            # Tập dữ liệu huấn luyện mẫu (5000 mẫu)
│   └── test.csv             # Tập dữ liệu kiểm thử mẫu (1000 mẫu)
├── train.sh                 # Script Bash tự động chuẩn bị dữ liệu & huấn luyện
├── inference.sh             # Script Bash tự động chạy kiểm thử
├── requirements.txt         # Các thư viện phụ thuộc
└── README.md                # Tài liệu hướng dẫn dự án


🛠️ 1. Cài đặt môi trường

Nên sử dụng máy có GPU NVIDIA (ví dụ: Google Colab T4/A100, Kaggle) để tận dụng tốc độ huấn luyện của Unsloth.

# Clone repository
git clone [https://github.com/QTuan211205/banking-intent-unsloth.git](https://github.com/QTuan211205/banking-intent-unsloth.git)
cd banking-intent-unsloth

# Cài đặt thư viện
pip install -r requirements.txt


📥 2. Chuẩn bị dữ liệu

Tải tập dữ liệu BANKING77 và lấy mẫu:

python scripts/preprocess_data.py


🧠 3. Huấn luyện mô hình

Chạy script để tinh chỉnh mô hình:

bash train.sh


📦 4. Tải trọng số (Pre-trained Weights)

Để chạy suy luận mà không cần huấn luyện lại, hãy thực hiện lệnh sau trong Notebook để tải mô hình từ Google Drive:

# Cài đặt gdown và tải thư mục mô hình
!pip install gdown
!gdown --folder 1RSHsvAPPlxq6fiXD8pal3JaX1DlDH9Pc -O /kaggle/working/model_weights


Lưu ý: Đảm bảo file configs/inference.yaml trỏ vào /kaggle/working/model_weights.

🎯 5. Chạy suy luận (Inference)

Thực thi script suy luận để dự đoán ý định:

bash inference.sh


💡 Usage Example:

from scripts.inference import IntentClassification

# Khởi tạo với file cấu hình
classifier = IntentClassification(model_path="configs/inference.yaml")

# Dự đoán
message = "I lost my credit card, what should I do?"
print(f"🤖 Dự đoán: {classifier(message)}")


🎯 Kết quả: Độ chính xác (Accuracy) trên tập kiểm thử đạt 91.40%.