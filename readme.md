# Smart Grid DApp

## Overview

This decentralized application (DApp) leverages the power of federated learning and blockchain technology to enable secure and collaborative data analysis for smart grids. 

## Key Features

- **Federated Learning:** Train a shared machine learning model across multiple smart grids without directly sharing raw data.
- **Differential Privacy:** Enhance data privacy during model training using PySyft's DPAdam optimizer.
- **Homomorphic Encryption:**  Explore secure data aggregation techniques using homomorphic encryption.
- **Blockchain Integration:** Record model updates and data usage agreements on the Energy Web Chain for transparency and auditability.

## Technologies Used

- **PySyft:**  A framework for secure and private federated learning.
- **PyTorch:** A deep learning library for model development.
- **Flask:**  A Python web framework for building the backend API.
- **Web3.py:** A library for interacting with the Energy Web Chain.
- **Solidity:** The programming language for writing the smart contract.
- **Truffle:** A development environment for Ethereum and EVM-compatible blockchains.
- **Ganache:** A local blockchain emulator for development and testing.

## Project Structure
smart-grid-dapp/
├── contracts
│   └── DataSharing.sol
├── migrations
│   ├── 2_deploy_contracts.js
│   └── 1_initial_migration.js
├── models
│   ├── handler
│   │   └── custom_handler.py
│   └── model.py
├── static
│   ├── js
│   │   └── scripts.js
│   └── css
│       └── styles.css
├── templates
│   └── index.html
├── utils
│   ├── blockchain.py
│   └── prediction.py
├── tests
│   ├── test_blockchain.py
│   └── test_prediction.py
├── setup
│   ├── setup_python.sh
│   ├── setup_torchserve.sh
│   └── setup_node.sh
├── app.py
├── federated_learning.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml

## Getting Started

### Prerequisites

- **Node.js:** [https://nodejs.org/](https://nodejs.org/)
- **Python 3.8+:** [https://www.python.org/](https://www.python.org/)
- **Docker:** [https://www.docker.com/](https://www.docker.com/)
- **Energy Web Chain Account:** [https://www.energyweb.org/](https://www.energyweb.org/)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/smart-grid-dapp.git
   cd smart-grid-dapp

2. **Install Dependencies:**

# Install Node.js dependencies
bash setup/setup_node.sh

# Create a Python virtual environment and activate it
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install TorchServe
bash setup/setup_torchserve.sh

## Configuration

### Blockchain Configuration:

1. Open `truffle-config.js`.
2. Configure the `networks` section to connect to the Energy Web Chain.
3. Provide your EWC account credentials and an API key for an EWC node provider (if necessary).

### Private Key:

**Important:** Replace `"YOUR_PRIVATE_KEY"` in `app.py` and `test_blockchain.py` with a valid private key for interacting with the blockchain. For security reasons, use a test account and a testnet for development. 

## Running the Application

1. **Start Ganache (for local development):**
   ```bash
   ganache-cli
   ```

2. **Deploy the Smart Contract:**
   ```bash
   truffle compile
   truffle migrate --network ewc 
   ```

3. **Train the Model:**
   ```bash
   python federated_learning.py
   ```

4. **Archive and Serve the Model:**
   ```bash
   torch-model-archiver --model-name smartgridmodel --version 1.0 \
   --model-file models/model.py --serialized-file global_model.pth \
   --handler models/handler/custom_handler.py --export-path model_store

   torchserve --start --ncs --model-store model_store --models smartgridmodel=smartgridmodel.mar
   ```

5. **Run the Flask Application:**
   ```bash
   python app.py
   ```

6. **Access the UI:**
   Open the provided URL (from Codespaces or your local server) in your web browser.

## Running Tests

* **Run All Tests:**
   ```bash
   python -m unittest discover
   ```

* **Run Specific Tests:**
   ```bash
   python -m unittest tests/test_blockchain.py
   python -m unittest tests/test_prediction.py
   ```

## Future Work

* Integrate real smart grid data sources.
* Develop a more sophisticated user interface.
* Implement additional blockchain functionalities (e.g., data usage agreements, tokenized rewards).
* Explore more advanced privacy-preserving techniques.
* Deploy to a production environment.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests.

## License

This project is licensed under the MIT License.