import os
import sys
import logging

import flwr as fl
import torch
from torch.utils.data import DataLoader
from typing import Optional
try:
    from opacus import PrivacyEngine  # type: ignore
except Exception:  # pragma: no cover
    PrivacyEngine = None  # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.lora_model import get_lora_model
from utils.dataset import ToyMentalHealthDataset, ChatLogDataset


logging.basicConfig(level=logging.INFO)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


USE_CHAT_DATA = os.environ.get("USE_CHAT_DATA", "0") == "1"
USE_DP = os.environ.get("USE_DP", "0") == "1"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
NOISE_MULTIPLIER = float(os.environ.get("NOISE_MULTIPLIER", "1.0"))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", "1.0"))


# Prepare model
model = get_lora_model().to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


# Dataset selection: chat logs (tokenized) or toy synthetic
if USE_CHAT_DATA:
    dataset = ChatLogDataset()
    if len(dataset) == 0:
        logging.warning("ChatLogDataset is empty; falling back to ToyMentalHealthDataset")
        dataset = ToyMentalHealthDataset(size=50)
else:
    dataset = ToyMentalHealthDataset(size=50)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# Attach Opacus privacy engine conditionally (avoid incompatibility issues)
def maybe_attach_privacy(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
) -> DataLoader:
    if not USE_DP:
        logging.info("Differential privacy disabled (USE_DP=0). Proceeding without Opacus.")
        return loader
    if PrivacyEngine is None:
        logging.warning("Opacus not available; proceeding without differential privacy.")
        return loader
    try:
        privacy_engine = PrivacyEngine()
        m, opt, dl = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=NOISE_MULTIPLIER,
            max_grad_norm=MAX_GRAD_NORM,
        )
        # Update references if Opacus wraps them
        globals()["model"] = m
        globals()["optimizer"] = opt
        logging.info("Opacus privacy engine attached successfully.")
        return dl
    except Exception as e:
        logging.warning(
            "Failed to attach Opacus privacy engine (%s). Training without DP.", e
        )
        return loader

train_loader = maybe_attach_privacy(model, optimizer, train_loader)


def train():
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        if "attention_mask" in batch:
            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )
        else:
            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Training batch, loss={loss.item():.4f}")


def get_weights():
    """
    Get model weights as NumPy arrays.
    
    Note: CPU conversion is necessary for Flower's NumPyClient protocol.
    This only happens during parameter exchange (not during training),
    so the performance impact is minimal.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(weights):
    """Set model weights from NumPy arrays, moving them to the appropriate device."""
    state_dict = dict(zip(model.state_dict().keys(), [torch.tensor(w).to(DEVICE) for w in weights]))
    model.load_state_dict(state_dict, strict=True)


def evaluate_loader(loader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if "attention_mask" in batch:
                outputs = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    labels=batch["labels"].to(DEVICE),
                )
            else:
                outputs = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    labels=batch["labels"].to(DEVICE),
                )
            bs = batch["input_ids"].size(0)
            total_loss += outputs.loss.item() * bs
            total += bs
    return total_loss / max(total, 1)


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_weights()

    def fit(self, parameters, config):
        set_weights(parameters)
        train()
        return get_weights(), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_weights(parameters)
        eval_loss = evaluate_loader(train_loader)
        return float(eval_loss), len(train_loader.dataset), {}


if __name__ == "__main__":
    server_address = os.environ.get("FL_SERVER_ADDRESS", "127.0.0.1:8080")
    print(f"🤝 Starting Flower client, connecting to {server_address} ...")
    fl.client.start_client(
        server_address=server_address,
        client=FlowerClient().to_client(),
    )
