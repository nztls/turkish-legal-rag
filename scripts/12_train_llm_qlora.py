import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5 with QLoRA on Turkish legal QA SFT data."
    )

    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("outputs/llm_sft_data/train_sft.jsonl"),
    )

    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path("outputs/llm_sft_data/val_sft.jsonl"),
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/models/qwen25_legal_qlora"),
    )

    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Precision selection:
    # If the GPU supports BF16, use BF16.
    # Otherwise use FP16. This prevents BF16/FP16 GradScaler conflicts.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("CUDA available:", torch.cuda.is_available())
    print("BF16 supported:", use_bf16)
    print("Compute dtype:", compute_dtype)

    print("Base model:", args.base_model)
    print("Train file:", args.train_file)
    print("Val file:", args.val_file)
    print("Output dir:", args.output_dir)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Gradient accumulation:", args.grad_accum)
    print("Learning rate:", args.learning_rate)
    print("Max sequence length:", args.max_seq_length)

    if not args.train_file.exists():
        raise FileNotFoundError(f"Train file not found: {args.train_file}")

    if not args.val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {args.val_file}")

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.train_file),
            "validation": str(args.val_file),
        },
    )

    print("Train examples:", len(dataset["train"]))
    print("Validation examples:", len(dataset["validation"]))

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    # Required/recommended for QLoRA training on 4-bit loaded models.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=not use_bf16,
        bf16=use_bf16,
        max_grad_norm=0.0,
        max_length=args.max_seq_length,
        packing=False,
        dataset_text_field="text",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()

    adapter_dir = args.output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    print("QLoRA adapter saved to:", adapter_dir)


if __name__ == "__main__":
    main()