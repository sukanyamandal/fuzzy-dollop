import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import syft as sy
import torch.nn.functional as F


# --- Correct Import and Setup for VirtualWorker in PySyft 0.8.6 ---
sy.requires(">=0.8.6,<0.8.7")

# Start the domain
domain = sy.orchestra.launch(
    name="my-domain",
    port=8080,
    create_producer=True,
    n_consumers=3,
    dev_mode=True,
    reset=True,
)

# Create virtual workers associated with the domain
domain_client = sy.login(url="http://localhost:8080", email="info@openmined.org", password="changethis")
worker_names = [f"worker{i+1}" for i in range(3)] 
workers = [sy.VirtualWorker(hook=domain_client.hook, id=worker_name) for worker_name in worker_names]

# --- 1. Define the Neural Network Model ---
class SimpleNN(nn.Module):
    """
    A simple neural network with two fully connected layers.
    This model is designed for demonstration purposes and can be replaced
    with a more complex architecture for real-world applications.
    """

    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        """Defines the forward pass of the network."""
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. Helper Functions for Federated Learning ---
def train_local_model(model, local_data, epochs=2):
    """
    Trains a local model on a given dataset.

    Args:
        model (nn.Module): The neural network model to train.
        local_data (DataLoader): The local data to train the model on.
        epochs (int, optional): Number of local training epochs. Defaults to 2.

    Returns:
        state_dict: The trained state dictionary of the local model.
    """

    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        for data, target in local_data:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    return model.state_dict()

def aggregate_models(global_model, local_models):
    """
    Aggregates the weights of multiple local models into a global model.

    Args:
        global_model (nn.Module): The global model to update.
        local_models (list): A list of state dictionaries from the trained local models.

    Returns:
        dict: The updated state dictionary of the global model.
    """

    global_dict = copy.deepcopy(local_models[0])
    for key in global_dict.keys():
        for i in range(1, len(local_models)):
            global_dict[key] += local_models[i][key]
        global_dict[key] = torch.div(global_dict[key], len(local_models))
    return global_dict

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    # --- Simulate Data from Multiple Smart Grids ---
    torch.manual_seed(0)
    num_grids = 3
    data_per_grid = 200
    num_features = 3

    grid_data = []

    for i in range(num_grids):
        X = torch.randn(data_per_grid, num_features)
        y = (
            0.5 * X[:, 0]
            + 0.3 * X[:, 1]
            - 0.2 * X[:, 2]
            + torch.randn(data_per_grid).unsqueeze(1)
        )
        grid_data.append(TensorDataset(X, y))

    # --- Create Federated Data Loaders ---
    batch_size = 32

    federated_dataloaders = [
        sy.FederatedDataLoader(
            DataLoader(grid_data[i], batch_size=batch_size, shuffle=True),
            batch_size=batch_size,
            shuffle=True,
            worker=workers[i],
        )
        for i in range(num_grids)
    ]

    # --- Federated Training ---
    global_model = SimpleNN()
    num_rounds = 5

    for round in range(num_rounds):
        local_models = []
        for worker_data in federated_dataloaders:
            # Send the model to the worker
            local_model = copy.deepcopy(global_model).send(worker_data.worker)

            # --- Differential Privacy (Using DPAdam) ---
            optimizer = optim.DPAdam(
                params=local_model.parameters(),
                lr=0.01,
                eps=1.0,
                delta=1e-5,
                max_grad_norm=1.0,
            )

            # Train the local model using DPAdam
            local_model.train()
            criterion = nn.MSELoss()
            for epoch in range(2):
                for data, target in worker_data:
                    optimizer.zero_grad()
                    output = local_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

            # Get the trained model back from the worker
            local_model = local_model.get()
            local_models.append(local_model.state_dict())

        # Aggregate and update the global model
        global_state = aggregate_models(global_model, local_models)
        global_model.load_state_dict(global_state)

        print(f"Round {round+1} completed.")

    # --- Save the Trained Global Model ---
    torch.save(global_model.state_dict(), "global_model.pth")

    # --- Blockchain Interaction (Update Model Hash) ---
    from utils.blockchain import Blockchain
    blockchain = Blockchain()
    model_hash = "YOUR_MODEL_HASH" # Calculate the hash of your model
    tx_hash = blockchain.update_model("YOUR_PRIVATE_KEY", model_hash) 
    print(f"Model hash updated on blockchain. Transaction hash: {tx_hash}")

    # --- Homomorphic Encryption (Example) ---
    meter1_data = torch.tensor([2.5, 3.0, 2.8])
    meter2_data = torch.tensor([1.8, 2.2, 2.1])

    bob = domain.create_worker(name="bob")
    alice = domain.create_worker(name="alice")

    meter1_data_enc = meter1_data.fix_precision().share(
        bob, alice, crypto_provider=bob
    )
    meter2_data_enc = meter2_data.fix_precision().share(
        bob, alice, crypto_provider=bob
    )

    total_consumption_enc = (
        meter1_data_enc + meter2_data_enc
    )

    total_consumption = total_consumption_enc.get().float_precision()
    print("Total Encrypted Consumption:", total_consumption)