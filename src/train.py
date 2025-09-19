#!/usr/bin/env python3

import torch
import os
from datasets import load_dataset, Audio, DatasetDict
from transformers import (
    VoxtralForConditionalGeneration,
    VoxtralProcessor,
    Trainer,
    TrainingArguments,
    WhisperForConditionalGeneration
)
from peft import LoraConfig, get_peft_model
import wandb 


class VoxtralDataCollator:
    """Data collator for Voxtral STT training - processes audio and text."""
    
    def __init__(self, processor:VoxtralProcessor, model_id):
        self.processor = processor
        self.model_id = model_id
        self.pad_id = processor.tokenizer.pad_token_id

    def __call__(self, features):
        """
        Each feature should have:
          - "audio": raw audio (whatever your processor expects)
          - "text":  transcription string
        """
        texts  = [f["text"] for f in features]
        audios = [f["audio"]["array"] for f in features]

        # 1) Build the PROMPT part: [AUDIO]…[AUDIO] <transcribe>
        prompt = self.processor.apply_transcription_request(  # (same method you used)
            language="da",
            model_id=self.model_id if hasattr(self, "model_id") else None,
            audio=audios,
            format=["WAV"] * len(audios),
            return_tensors="pt",
        )
        # prompt["input_ids"]: shape [B, L_prompt]
        # keep any extra fields (e.g., audio features) to pass through to the model
        passthrough = {k: v for k, v in prompt.items()
                       if k not in ("input_ids", "attention_mask")}

        prompt_ids = prompt["input_ids"]           # [B, Lp]
        prompt_attn = prompt["attention_mask"]     # [B, Lp]
        B = prompt_ids.size(0)

        tok = self.processor.tokenizer
        # 2) Tokenize transcriptions WITHOUT padding; we'll pad after concatenation

        text_tok = tok(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=256,
            return_tensors=None,
        )
        text_ids_list = text_tok["input_ids"]

        # 3) Concatenate: input_ids = [PROMPT] + [TEXT]
        input_ids, attention_mask, labels = [], [], []
        for i in range(B):
            p_ids = prompt_ids[i].tolist()
            p_att = prompt_attn[i].tolist()
            t_ids = text_ids_list[i]


            ids  = p_ids + t_ids + [tok.eos_token_id]
            attn = p_att + [1] * (len(t_ids) + 1) 
            # labels: mask prompt tokens, learn only on text tokens
            lab  = [-100] * len(p_ids) + t_ids + [tok.eos_token_id]

            input_ids.append(ids)
            attention_mask.append(attn)
            labels.append(lab)

        # 4) Pad to max length in batch
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        max_len = max(len(x) for x in input_ids)

        def pad_to(seq, fill, L):
            return seq + [fill] * (L - len(seq))

        input_ids      = [pad_to(x, pad_id, max_len) for x in input_ids]
        attention_mask = [pad_to(x, 0,      max_len) for x in attention_mask]
        labels         = [pad_to(x, -100,   max_len) for x in labels]
        
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        # 5) Include processor outputs needed by the model (e.g., audio features)
        for k, v in passthrough.items():
            batch[k] = v

        return batch
    
def load_and_prepare_dataset():
    """Load and prepare CoRal dataset for training."""
    dataset_name = "CoRal-project/coral"
    dataset_config = "read_aloud"
    
    print(f"Loading dataset: {dataset_name}/{dataset_config}")
    
    # Load train and validation splits directly
    train_dataset = load_dataset(dataset_name, dataset_config, split="train")
    eval_dataset = load_dataset(dataset_name, dataset_config, split="val")
    
    # Cast audio to 16kHz (required for Voxtral)
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    return train_dataset, eval_dataset

def main():
    # --- 1. Load Models and Override Weights ---

    os.environ["WANDB_PROJECT"] = "voxtral-finetune-coral-large"

    print("Loading Voxtral and Whisper models to override weights...")
    voxtral_model_checkpoint = "mistralai/Voxtral-Small-24B-2507"
    whisper_model_checkpoint = "CoRal-project/roest-whisper-large-v1"
    
    
    # Load Voxtral and Whisper models
    model = VoxtralForConditionalGeneration.from_pretrained(
        voxtral_model_checkpoint,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="flash_attention_2"
    )


    ## Optional override audio encoder weights
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        whisper_model_checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    whisper_encoder_state_dict = whisper_model.model.encoder.state_dict()
    model.audio_tower.load_state_dict(whisper_encoder_state_dict)
    print("Voxtral's audio tower weights have been successfully overridden.")

    ## Optional - dont train audio encoder
    for param in model.audio_tower.parameters():
       param.requires_grad = False

    config = LoraConfig( 
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head","linear_1","linear_2"],
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, config) 
    model.print_trainable_parameters()
    
    train_dataset, eval_dataset = load_and_prepare_dataset()
    data_collator = VoxtralDataCollator(VoxtralProcessor.from_pretrained(voxtral_model_checkpoint), voxtral_model_checkpoint)
    
    training_args = TrainingArguments(
        output_dir="./danstral-finetune",
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=2,
        bf16=True,
        logging_steps=10,
        eval_steps=1000 if eval_dataset else None,
        save_steps=2000,
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        report_to="wandb", 
        remove_unused_columns=False,
        dataloader_num_workers=8,
        ddp_find_unused_parameters=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()

    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    data_collator.processor.save_pretrained(training_args.output_dir)
    
    if eval_dataset:
        results = trainer.evaluate()
        print(f"Final evaluation results: {results}")
    
    print("Training completed successfully!")
    wandb.finish()

if __name__ == "__main__":
    main()
