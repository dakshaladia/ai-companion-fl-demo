import flwr as fl
import logging
import logging
logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🚀 Starting Flower server on 0.0.0.0:8080 ...")

    strategy = fl.server.strategy.FedAvg()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    print("✅ Server finished training rounds")
