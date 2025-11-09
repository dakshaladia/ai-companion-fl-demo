import os
import sys
import flwr as fl
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Optional, Union
from flwr.common import Parameters, FitRes, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
import logging
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.lora_model import get_lora_model

logging.basicConfig(level=logging.INFO)

# Directory for saving FL checkpoints
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")


def save_model_checkpoint(parameters: List[np.ndarray], path: str):
    """Save aggregated model weights to checkpoint file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Load a fresh model to get the correct state dict structure
    model = get_lora_model()
    
    # Convert parameters (list of arrays) back to state dict
    state_dict = dict(zip(model.state_dict().keys(), [torch.tensor(w) for w in parameters]))
    
    # Save the checkpoint
    torch.save({
        'model_state_dict': state_dict,
        'model_type': 'lora_distilbert_mood_classifier',
        'num_labels': 2,
    }, path)
    print(f"✅ Saved model checkpoint to {path}")


class SaveModelStrategy(FedAvg):
    """Custom FedAvg strategy that saves the model after each round."""
    
    def __init__(self, checkpoint_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_path = checkpoint_path
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], dict]:
        """Aggregate fit results and save the model."""
        # Call parent to get aggregated parameters
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            # Convert Parameters (bytes) to list of NumPy arrays (Flower utility)
            parameters_list = parameters_to_ndarrays(aggregated_parameters)
            
            # Save checkpoint after each round
            round_path = self.checkpoint_path.replace(".pt", f"_round{server_round}.pt")
            save_model_checkpoint(parameters_list, round_path)
            
            # Also save as the "latest" checkpoint
            save_model_checkpoint(parameters_list, self.checkpoint_path)
            
            print(f"📊 Round {server_round} complete - model saved")
        
        return aggregated_parameters, aggregated_metrics


if __name__ == "__main__":
    # Allow overriding the bind address to avoid IPv6/IPv4 mismatches
    server_address = os.environ.get("FL_SERVER_ADDRESS", "127.0.0.1:8080")
    print(f"🚀 Starting Flower server on {server_address} ...")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "fl_mood_classifier.pt")

    strategy = SaveModelStrategy(
        checkpoint_path=checkpoint_path,
        min_fit_clients=2,
        min_available_clients=2,
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    print("✅ Server finished training rounds")
    print(f"💾 Final model saved to {checkpoint_path}")
