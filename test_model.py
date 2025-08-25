import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def test_finetuned_model():
    print("🧪 ファインチューニング済みモデルのテスト...")
    
    # デバイスの設定
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple Silicon GPU (MPS) を使用します")
    else:
        device = torch.device("cpu")
        print("💻 CPUを使用します")
    
    # ベースモデルとファインチューニング済みモデルのロード
    base_model_name = "cyberagent/open-calm-small"  # iOS向け軽量モデル
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,  # MPSはfloat32
        trust_remote_code=True
    ).to(device)
    
    # LoRAアダプターをロード
    model = PeftModel.from_pretrained(model, "./finetuned_model")
    tokenizer = AutoTokenizer.from_pretrained("./finetuned_model", trust_remote_code=True)
    
    # テスト質問
    test_questions = [
        "好きな食べ物は何？",
        "週末は何して過ごしたい？",
        "最近嬉しかったことは？",
        "私のこと好き？"
    ]
    
    print("\n📝 テスト結果:")
    print("-" * 50)
    
    for question in test_questions:
        prompt = f"質問: {question}\n回答: "
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # inputsをモデルと同じデバイスに移動
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("回答: ")[-1]
        
        print(f"Q: {question}")
        print(f"A: {answer}")
        print("-" * 50)

if __name__ == "__main__":
    test_finetuned_model()