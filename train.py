import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configuration
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
output_dir = "./results"
merged_dir = "./merged_model"
dataset_file = "train.jsonl"
MAX_SEQ_LENGTH = 1024

def train():
    print(f"Loading dataset from {dataset_file}")
    dataset = load_dataset("json", data_files=dataset_file, split="train")

    # BitsAndBytesConfig for 4-bit loading on T4
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    print(f"Loading model {model_id} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Format and tokenize the dataset
    def format_and_tokenize(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)

    # Training arguments — using plain TrainingArguments (works with every version)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        max_grad_norm=0.3,
        max_steps=200,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("Starting training...")
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=data_collator,
    )

    trainer.train()

    print("Saving PEFT adapter...")
    trainer.model.save_pretrained(os.path.join(output_dir, "adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "adapter"))

    print("Merge and save full model...")
    del model
    del trainer
    torch.cuda.empty_cache()

    # Reload base model in FP16 and merge the LoRA adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, os.path.join(output_dir, "adapter"))
    model = model.merge_and_unload()

    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    print(f"Model successfully saved to {merged_dir}")

if __name__ == "__main__":
    train()
