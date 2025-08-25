import torch
import torch.onnx
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np

def export_to_onnx():
    print("🔄 ONNX形式にエクスポート開始...")
    
    # デバイス設定
    device = torch.device("cpu")  # ONNXエクスポートはCPUで実行
    
    # モデルのロード
    print("📦 モデルをロード中...")
    base_model_name = "cyberagent/open-calm-small"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device)
    
    # LoRAアダプターをマージ
    model = PeftModel.from_pretrained(model, "./finetuned_model")
    model = model.merge_and_unload()  # LoRAをベースモデルにマージ
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("./finetuned_model", trust_remote_code=True)
    
    # 固定長の入力を準備
    print("📝 サンプル入力を準備中...")
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 20)).to(device)
    
    # ONNXにエクスポート（シンプルなforward passのみ）
    print("🔄 ONNXにエクスポート中...")
    
    # より単純なラッパーモデルを作成
    class SimpleModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, input_ids):
            # 単純にlogitsのみを返す
            outputs = self.model(input_ids, use_cache=False)
            return outputs.logits
    
    simple_model = SimpleModel(model)
    simple_model.eval()
    
    # ONNXエクスポート
    torch.onnx.export(
        simple_model,
        dummy_input,
        "model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size', 1: 'sequence'}
        }
    )
    
    print("✅ ONNXモデルを保存しました: model.onnx")
    
    # トークナイザーの語彙を保存
    import json
    vocab = tokenizer.get_vocab()
    with open("tokenizer_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # 特殊トークンも保存
    special_tokens = {
        "pad_token": tokenizer.pad_token,
        "eos_token": tokenizer.eos_token,
        "bos_token": tokenizer.bos_token if hasattr(tokenizer, 'bos_token') else None,
        "unk_token": tokenizer.unk_token if hasattr(tokenizer, 'unk_token') else None,
    }
    with open("special_tokens.json", "w", encoding="utf-8") as f:
        json.dump(special_tokens, f, ensure_ascii=False, indent=2)
    
    print("✅ トークナイザー情報を保存しました")
    print("\n📱 次のステップ:")
    print("1. onnx2tf でTensorFlow Liteに変換")
    print("2. Core MLToolsでCore MLに変換")
    print("3. iOSアプリに組み込み")

if __name__ == "__main__":
    export_to_onnx()