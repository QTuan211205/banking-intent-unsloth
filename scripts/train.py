import os
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# Hàm format dữ liệu dạng Hỏi-Đáp
def formatting_prompts_func(examples, tokenizer):
    prompt_template = """Phân loại ý định của khách hàng ngân hàng dựa trên câu truy vấn dưới đây.
Chỉ trả về chính xác tên ý định bằng tiếng Anh, không giải thích gì thêm.

### Truy vấn:
{}

### Ý định:
{}"""
    texts = examples["text"]
    intents = examples["intent_name"]
    formatted_texts = []
    
    for text, intent in zip(texts, intents):
        # Thêm EOS_TOKEN để mô hình biết điểm dừng
        text_with_prompt = prompt_template.format(text, intent) + tokenizer.eos_token
        formatted_texts.append(text_with_prompt)
        
    return { "formatted_text" : formatted_texts }

def main():
    # 1. Cấu hình (Thay thế cho file train.yaml để chạy độc lập dễ dàng hơn)
    CFG = {
        "model_name": "unsloth/llama-3-8b-bnb-4bit",
        "max_seq_length": 256,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "batch_size": 2,
        "grad_accumulation": 8,
        "epochs": 2,
        "seed": 3407,
        "output_dir": "../banking_outputs"
    }

    # 2. Tải dữ liệu đã tiền xử lý
    train_df = pd.read_csv("../sample_data/train.csv")
    train_dataset = Dataset.from_pandas(train_df)

    # 3. Load Model & Tokenizer bằng Unsloth
    print("Đang khởi tạo mô hình Llama-3...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = CFG["model_name"],
        max_seq_length = CFG["max_seq_length"],
        dtype = None,
        load_in_4bit = True,
    )

    # 4. Gắn LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = CFG["lora_r"],
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = CFG["lora_alpha"],
        lora_dropout = CFG["lora_dropout"],
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = CFG["seed"],
    )

    # 5. Format Dataset
    train_dataset = train_dataset.map(lambda x: formatting_prompts_func(x, tokenizer), batched = True)

    # 6. Thiết lập SFTTrainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        args = SFTConfig(
            output_dir = CFG["output_dir"],
            per_device_train_batch_size = CFG["batch_size"],
            gradient_accumulation_steps = CFG["grad_accumulation"],
            learning_rate = CFG["learning_rate"],
            num_train_epochs = CFG["epochs"],
            logging_steps = 10,
            optim = "paged_adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = CFG["seed"],
            report_to = "none",
            dataset_text_field = "formatted_text",
            max_seq_length = CFG["max_seq_length"],
            dataset_num_proc = 2,
            packing = False,
            padding_free = False,
        ),
    )

    # 7. Bắt đầu Train
    print("🚀 Bắt đầu quá trình Fine-tuning...")
    trainer.train()

    # 8. Lưu Model
    model.save_pretrained(CFG["output_dir"])
    tokenizer.save_pretrained(CFG["output_dir"])
    print(f"✅ Đã lưu mô hình thành công tại: {CFG['output_dir']}")

if __name__ == "__main__":
    main()