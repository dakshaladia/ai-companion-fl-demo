import os
import json
from typing import Optional, List, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ToyMentalHealthDataset(Dataset):
    def __init__(self, size=100):
        self.data = [("I feel happy", 1), ("I feel sad", 0)] * (size // 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        return {"input_ids": torch.randint(0, 1000, (16,)), "labels": torch.tensor(label)}


class ChatLogDataset(Dataset):
    """Reads locally stored chats and yields tokenized samples for classification.

    Each user message may optionally include a label (0/1). Unlabeled entries are skipped
    by default unless include_unlabeled=True, in which case label=-100 is used so loss is
    ignored for those positions.
    """

    def __init__(
        self,
        chat_log_path: Optional[str] = None,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        include_unlabeled: bool = False,
    ) -> None:
        root_dir = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(root_dir, "data")
        self.chat_log_path = chat_log_path or os.path.join(data_dir, "chat.jsonl")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.include_unlabeled = include_unlabeled
        self.samples: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.chat_log_path):
            self.samples = []
            return
        with open(self.chat_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("role") != "user":
                    continue
                label = event.get("label", None)
                if label is None and not self.include_unlabeled:
                    continue
                text = (event.get("text") or "").strip()
                if not text:
                    continue
                self.samples.append({"text": text, "label": label})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        enc = self.tokenizer(
            sample["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        label = -100 if sample["label"] is None else int(sample["label"])  # ignore unlabeled
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }
