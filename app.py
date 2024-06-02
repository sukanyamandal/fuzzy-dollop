from flask import Flask, render_template, request, jsonify
from utils.blockchain import Blockchain
from utils.prediction import Prediction
from federated_learning import SimpleNN, train_local_model, aggregate_models
import torch
from torch.utils.data import DataLoader, TensorDataset
import json
import syft as sy
import torch.nn as nn
import torch.optim as optim

hook = sy.TorchHook(torch)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

blockchain = Blockchain()
prediction = Prediction()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    grid_data = request.json
    tx_hash = blockchain.register_grid(grid_data)
    return jsonify({'transaction_hash': tx_hash})

@app.route('/train', methods=['POST'])
def train_model():
    # ... (Federated learning code with DP)
    # --- Data Simulation (Same as in federated_learning.py) ---
    torch.manual_seed(0)
    num_grids = 3
    data_per_grid = 200
    num_features = 3

    grid_data = []
    for i in range(num_grids):
        X = torch.randn(data_per_grid, num_features)
        y = (0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + torch.randn(data_per_grid).unsqueeze(1))
        grid_data.append(TensorDataset(X, y))

    # --- Create Federated Data Loaders ---
    batch_size = 32
    workers = [sy.VirtualWorker(hook, id=f"worker{i+1}") for i in range(num_grids)]
    federated_dataloaders = [
        sy.FederatedDataLoader(
            DataLoader(grid_data[i], batch_size=batch_size, shuffle=True),
            batch_size=batch_size,
            shuffle=True,
            worker=workers[i],
        )
        for i in range(num_grids)
    ]

    # --- Federated Training (Using DPAdam) ---
    global_model = SimpleNN()
    num_rounds = 5

    for round in range(num_rounds):
        local_models = []
        for worker_data in federated_dataloaders:
            # Send the global model to the worker
            local_model = SimpleNN().copy(global_model).send(worker_data.worker)

            # --- Differential Privacy (Using DPAdam) ---
            optimizer = optim.DPAdam(
                params=local_model.parameters(),
                lr=0.01,
                eps=1.0,  # Privacy budget - adjust as needed
                delta=1e-5,
                max_grad_norm=1.0
            )

            # Train the local model using DPAdam
            local_model.train()
            criterion = nn.MSELoss()
            for epoch in range(2):  # Local epochs
                for data, target in worker_data:
                    optimizer.zero_grad()
                    output = local_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

            # Get the trained model back
            local_models.append(local_model.get().state_dict())

        # Aggregate and update the global model
        global_state = aggregate_models(global_model, local_models)
        global_model.load_state_dict(global_state)

        print(f"Round {round+1} completed.")

    # --- Save the Trained Global Model ---
    torch.save(global_model.state_dict(), "global_model.pth")

    # --- Blockchain Interaction (Update Model Hash) ---
    model_hash = "YOUR_MODEL_HASH"  # Calculate the hash of your model
    tx_hash = blockchain.update_model("YOUR_PRIVATE_KEY", model_hash)
    print(f"Model hash updated on blockchain. Transaction hash: {tx_hash}")

    return jsonify({"message": "Model training initiated and hash updated on blockchain."})

@app.route('/predict', methods=['POST'])
def predict():
    # ... (Prediction code)
    input_data = request.get_json(force=True)
    input_tensor = torch.tensor(input_data['input'], dtype=torch.float32)

    global_model = SimpleNN()
    global_model.load_state_dict(torch.load("global_model.pth"))
    global_model.eval()

    with torch.no_grad():
        prediction = global_model(input_tensor)

    return jsonify({'prediction': prediction.item()})

@app.route('/aggregate_consumption', methods=['POST'])
def aggregate_consumption():
    # ... (Homomorphic Encryption code)
    """
    Example route to demonstrate homomorphic encryption for aggregating 
    smart meter data from different grids.
    """
    try:
        # --- Receive Encrypted Data from Smart Grids ---
        # In a real application, you would receive this data securely from the grids.
        # For this example, we'll simulate it. 
        data = request.get_json(force=True)
        meter1_data_enc = torch.tensor(data['meter1'], dtype=torch.float32)
        meter2_data_enc = torch.tensor(data['meter2'], dtype=torch.float32)

        # --- Create Virtual Workers (representing Smart Grids) ---
        bob = sy.VirtualWorker(hook, id="bob")  # Smart Grid 1
        alice = sy.VirtualWorker(hook, id="alice")  # Smart Grid 2

        # --- Reconstruct Shared Tensors from Received Data ---
        meter1_data = sy.FloatTensor.from_pointer(meter1_data_enc, owner=bob).share(bob, alice, crypto_provider=bob)
        meter2_data = sy.FloatTensor.from_pointer(meter2_data_enc, owner=alice).share(bob, alice, crypto_provider=bob)

        # --- Perform Encrypted Aggregation ---
        total_consumption_enc = meter1_data + meter2_data

        # --- Decrypt and Get the Result ---
        total_consumption = total_consumption_enc.get().float_precision()

        return jsonify({'total_consumption': total_consumption.item()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)