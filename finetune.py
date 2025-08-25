import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import os

def load_conversation_data(file_path):
    conversations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = f"質問: {data['instruction']}\n回答: {data['output']}"
            conversations.append({"text": text})
    return conversations

def main():
    print("🚀 ファインチューニングを開始します...")
    
    # デバイスの設定（M4 Mac用）
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple Silicon GPU (MPS) を使用します")
    else:
        device = torch.device("cpu")
        print("💻 CPUを使用します")
    
    # モデルとトークナイザーの設定
    # iOS向けに最適化されたモデルを選択
    model_name = "cyberagent/open-calm-small"  # 160M（iOS最適・推奨）
    # model_name = "rinna/japanese-gpt2-small"  # 110M（最軽量）
    # model_name = "rinna/japanese-gpt2-medium"  # 336M（品質重視）
    # model_name = "microsoft/Phi-3-mini-4k-instruct"  # 3.8B（iOSには大きすぎる）
    
    print(f"📦 モデル {model_name} をロード中...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # MPSはfloat32を使用
        trust_remote_code=True
    ).to(device)
    
    # LoRA設定
    print("⚙️ LoRA設定を適用中...")
    
    # モデルのモジュール名を確認
    print("📋 利用可能なモジュール:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # リーフモジュールのみ
            print(f"  - {name}: {module.__class__.__name__}")
            if "query" in name or "key" in name or "value" in name or "dense" in name:
                break
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        # OpenCALM用のターゲットモジュール（自動検出されるデフォルトを使用）
        target_modules=None  # Noneにすると自動的に適切なモジュールが選択される
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # データセットの準備
    print("📚 データセットを準備中...")
    conversations = load_conversation_data("conversation_data.jsonl")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )
    
    dataset = Dataset.from_list(conversations)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=10,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        use_mps_device=torch.backends.mps.is_available(),  # MPS使用
        fp16=False,  # MPSではfp16は使用しない
        push_to_hub=False,
    )
    
    # データコレーターの設定
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # トレーナーの設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # トレーニング開始
    print("🏋️ トレーニング開始...")
    trainer.train()
    
    # モデルの保存
    print("💾 モデルを保存中...")
    model.save_pretrained("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")
    
    print("✨ ファインチューニング完了！")

if __name__ == "__main__":
    main()