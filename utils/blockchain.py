from web3 import Web3
import json

class Blockchain:
    def __init__(self):
        self.web3 = Web3(Web3.HTTPProvider('https://rpc.energyweb.org'))
        self.contract = self.load_contract()

    def load_contract(self):
        with open('contracts/DataSharing.json') as f:
            contract_data = json.load(f)
        abi = contract_data['abi']
        address = Web3.toChecksumAddress(contract_data['networks']['246']['address']) # Update network ID
        return self.web3.eth.contract(address=address, abi=abi)

    def register_grid(self, grid_data):
        account = self.web3.eth.account.from_key(grid_data['private_key'])
        nonce = self.web3.eth.getTransactionCount(account.address)
        tx = self.contract.functions.registerSmartGrid(grid_data['name']).buildTransaction({
            'chainId': 246, # EWC Chain ID
            'gas': 700000,
            'gasPrice': self.web3.toWei('1', 'gwei'),
            'nonce': nonce,
        })
        signed_tx = self.web3.eth.account.sign_transaction(tx, grid_data['private_key'])
        tx_hash = self.web3.eth.sendRawTransaction(signed_tx.rawTransaction)
        return tx_hash.hex()

    def update_model(self, private_key, model_hash):
        account = self.web3.eth.account.from_key(private_key)
        nonce = self.web3.eth.getTransactionCount(account.address)
        tx = self.contract.functions.updateModel(model_hash).buildTransaction({
            'chainId': 246,
            'gas': 700000,
            'gasPrice': self.web3.toWei('1', 'gwei'),
            'nonce': nonce,
        })
        signed_tx = self.web3.eth.account.sign_transaction(tx, private_key)
        tx_hash = self.web3.eth.sendRawTransaction(signed_tx.rawTransaction)
        return tx_hash.hex()