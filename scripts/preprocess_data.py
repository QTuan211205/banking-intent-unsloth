import os
import pandas as pd
from datasets import load_dataset

def main():
    print("Đang tải dữ liệu banking77 từ Hugging Face...")
    raw_dataset = load_dataset("banking77")

    # Lấy danh sách tên nhãn
    label_names = raw_dataset["train"].features["label"].names
    id2label = {i: label for i, label in enumerate(label_names)}

    # Chuyển sang DataFrame của Pandas
    train_df_full = raw_dataset["train"].to_pandas()
    test_df_full = raw_dataset["test"].to_pandas()

    # Map label ID sang tên ý định tiếng Anh
    train_df_full['intent_name'] = train_df_full['label'].map(id2label)
    test_df_full['intent_name'] = test_df_full['label'].map(id2label)

    # Lấy mẫu ngẫu nhiên (Subsetting)
    SEED = 3407
    train_df_sampled = train_df_full.sample(n=5000, random_state=SEED)
    test_df_sampled = test_df_full.sample(n=1000, random_state=SEED)

    print(f"Đã lấy mẫu: Train ({len(train_df_sampled)}), Test ({len(test_df_sampled)})")

    # Tạo thư mục sample_data ở thư mục gốc (ngoài thư mục scripts) và lưu file
    os.makedirs("../sample_data", exist_ok=True)
    train_df_sampled.to_csv("../sample_data/train.csv", index=False)
    test_df_sampled.to_csv("../sample_data/test.csv", index=False)
    
    print("✅ Đã lưu dữ liệu vào thư mục ../sample_data/ thành công!")

if __name__ == "__main__":
    main()