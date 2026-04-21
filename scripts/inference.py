import yaml
import torch
from unsloth import FastLanguageModel

class IntentClassification:
    def __init__(self, model_path):
        """
        Khởi tạo và load mô hình dựa trên file cấu hình YAML.
        """
        # 1. Đọc file cấu hình YAML để lấy đường dẫn thực tế của model
        with open(model_path, 'r') as f:
            config = yaml.safe_load(f)
        
        checkpoint_path = config.get('model_checkpoint_path')
        max_seq_length = config.get('max_seq_length', 256)

        print(f"🚀 Đang tải trọng số từ: {checkpoint_path}")
        
        # 2. Load mô hình và tokenizer bằng Unsloth
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = checkpoint_path,
            max_seq_length = max_seq_length,
            dtype = None,
            load_in_4bit = True,
        )
        
        # Bật chế độ suy luận siêu tốc
        FastLanguageModel.for_inference(self.model)
        
        # Template prompt (giữ nguyên như lúc train)
        self.prompt_template = """Phân loại ý định của khách hàng ngân hàng dựa trên câu truy vấn dưới đây.
Chỉ trả về chính xác tên ý định bằng tiếng Anh, không giải thích gì thêm.

### Truy vấn:
{}

### Ý định:
"""
    
    def __call__(self, message):
        """
        Nhận vào tin nhắn và trả về nhãn ý định.
        """
        inputs = self.tokenizer(
            [self.prompt_template.format(message)], 
            return_tensors = "pt"
        ).to("cuda")
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens = 64, 
            use_cache = True, 
            pad_token_id = self.tokenizer.eos_token_id
        )
        
        decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
        
        # Trích xuất nhãn dự đoán
        predicted_label = decoded_output.split("### Ý định:\n")[-1].strip()
        
        return predicted_label

# ==========================================
# VÍ DỤ SỬ DỤNG (Tuân thủ yêu cầu đồ án)
# ==========================================
if __name__ == "__main__":
    # Truyền đường dẫn tới FILE CẤU HÌNH thay vì thư mục model
    CONFIG_FILE = "configs/inference.yaml" 
    
    classifier = IntentClassification(model_path=CONFIG_FILE)
    
    test_message = "I lost my credit card, what should I do?"
    print(f"\nKhách hàng hỏi: '{test_message}'")
    
    predicted_intent = classifier(test_message)
    print(f"🤖 Llama-3 dự đoán: [{predicted_intent}]")