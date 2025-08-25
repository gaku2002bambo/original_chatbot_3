import torch
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np

def convert_to_coreml():
    print("🍎 Core ML変換を開始します...")
    
    # モデルのロード
    base_model_name = "cyberagent/open-calm-small"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    # LoRAアダプターをロード
    model = PeftModel.from_pretrained(model, "./finetuned_model")
    model = model.merge_and_unload()  # LoRAをベースモデルにマージ
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("./finetuned_model", trust_remote_code=True)
    
    print("📦 モデルをトレースモードに変換中...")
    
    # サンプル入力を作成
    sample_text = "質問: こんにちは\n回答: "
    inputs = tokenizer(sample_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    # TorchScriptに変換
    traced_model = torch.jit.trace(model, input_ids)
    
    print("🔄 Core MLに変換中...")
    
    # Core MLに変換
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=input_ids.shape, dtype=np.int32)],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram"
    )
    
    # モデルを保存
    mlmodel.save("ChatModel.mlpackage")
    print("✅ Core MLモデルを保存しました: ChatModel.mlpackage")
    
    # トークナイザーの語彙を保存
    import json
    vocab = tokenizer.get_vocab()
    with open("tokenizer_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print("✅ トークナイザー語彙を保存しました: tokenizer_vocab.json")

if __name__ == "__main__":
    convert_to_coreml()