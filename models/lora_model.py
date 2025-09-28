import torch
from torch import nn
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

def get_lora_model(model_name="distilbert-base-uncased", num_labels=2):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    
    # LoRA Config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_lin", "v_lin"],  # attention projections
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )
    
    model = get_peft_model(base_model, lora_config)
    return model
