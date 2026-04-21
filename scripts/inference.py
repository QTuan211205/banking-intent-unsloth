from unsloth import FastLanguageModel

class IntentClassification:
    def __init__(self, model_path):
        """
        Khởi tạo và load mô hình đã fine-tune cùng với tokenizer.
        """
        print("Đang tải trọng số mô hình từ ổ cứng, vui lòng đợi...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = 256,
            dtype = None,
            load_in_4bit = True,
        )
        # Bật chế độ suy luận siêu tốc của Unsloth
        FastLanguageModel.for_inference(self.model)
        
        # Mẫu prompt phải giống hệt lúc train
        self.prompt_template = """Phân loại ý định của khách hàng ngân hàng dựa trên câu truy vấn dưới đây.
Chỉ trả về chính xác tên ý định bằng tiếng Anh, không giải thích gì thêm.

### Truy vấn:
{}

### Ý định:
"""
    
    def __call__(self, message):
        """
        Nhận vào câu hỏi khách hàng và trả về nhãn ý định duy nhất.
        """
        # Format câu hỏi vào template
        inputs = self.tokenizer(
            [self.prompt_template.format(message)], 
            return_tensors = "pt"
        ).to("cuda")
        
        # Mô hình bắt đầu sinh chữ
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens = 64, 
            use_cache = True, 
            pad_token_id = self.tokenizer.eos_token_id
        )
        
        # Giải mã mảng số thành chữ
        decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
        
        # Cắt gọt chuỗi để chỉ lấy phần nhãn ý định nằm sau chữ "Ý định:\n"
        predicted_label = decoded_output.split("### Ý định:\n")[-1].strip()
        
        return predicted_label

# ==========================================
# VÍ DỤ SỬ DỤNG (Usage Example)
# ==========================================
if __name__ == "__main__":
    # Đường dẫn trỏ tới thư mục chứa file model (.safetensors) và config mà bạn đã lưu
    MODEL_CHECKPOINT = "../banking_outputs" 
    
    # 1. Khởi tạo class
    classifier = IntentClassification(model_path=MODEL_CHECKPOINT)
    print("✅ Đã load xong mô hình!")
    
    # 2. Tạo một câu hỏi giả lập
    test_message = "I lost my credit card, what should I do?"
    print(f"\nKhách hàng hỏi: '{test_message}'")
    
    # 3. Gọi mô hình dự đoán (sử dụng hàm __call__)
    predicted_intent = classifier(test_message)
    print(f"🤖 Llama-3 dự đoán ý định là: [{predicted_intent}]")