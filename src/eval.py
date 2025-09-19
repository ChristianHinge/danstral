from transformers import VoxtralForConditionalGeneration, AutoProcessor, WhisperForConditionalGeneration
import torch
from peft import PeftModel
from datasets import load_dataset, Audio
import os


voxtral_model_checkpoint = "mistralai/Voxtral-Small-24B-2507"
whisper_model_checkpoint = "CoRal-project/roest-whisper-large-v1"
lora_checkpoint = "hinge/danstral"


processor = AutoProcessor.from_pretrained(voxtral_model_checkpoint)
model = VoxtralForConditionalGeneration.from_pretrained(
    voxtral_model_checkpoint, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    attn_implementation="flash_attention_2"
)

whisper_model = WhisperForConditionalGeneration.from_pretrained(
    whisper_model_checkpoint,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

whisper_encoder_state_dict = whisper_model.model.encoder.state_dict()
model.audio_tower.load_state_dict(whisper_encoder_state_dict)
model = PeftModel.from_pretrained(model, lora_checkpoint)

coral = load_dataset("CoRal-project/coral", "read_aloud")
coral = coral.cast_column("audio", Audio(sampling_rate=16000))

results = []

for i in range(len(coral["test"])):
    sample = coral["test"][i]
    audio_data = sample['audio']
    ground_truth = sample['text']
    
    inputs = processor.apply_transcription_request(language="da", audio=audio_data['array'], format=["WAV"], model_id=voxtral_model_checkpoint)
    inputs = inputs.to("cuda:0", dtype=torch.bfloat16)
    
    outputs = model.generate(**inputs, max_new_tokens=256,do_sample=False)
    decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    print(f"Ground Truth: {ground_truth}")
    print(f"Prediction:   {decoded_outputs[0]}")
    print("-" * 40)
