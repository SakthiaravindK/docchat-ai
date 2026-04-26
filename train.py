import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

MODEL_NAME = "sshleifer/tiny-gpt2"
DATA_PATH = "data/train_dataset.jsonl"
OUTPUT_DIR = "models/docchat-lora"


def load_jsonl(path, limit=30):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def format_prompt(row):
    return f"""### Instruction:
{row['instruction']}

### Input:
{row['input']}

### Response:
{row['output']}"""


def main():
    print("Loading dataset...")
    rows = load_jsonl(DATA_PATH, limit=30)

    texts = [{"text": format_prompt(row)} for row in rows]
    dataset = Dataset.from_list(texts)

    print(f"Training samples: {len(dataset)}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=10,
        save_total_limit=1,
        fp16=False,
        report_to="none",
        no_cuda=True
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=256,
        args=training_args
    )

    print("Training started...")
    trainer.train()

    print("Saving LoRA adapter...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training completed.")
    print(f"Model saved at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()