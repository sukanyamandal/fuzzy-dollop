import unittest
from utils.blockchain import Blockchain

class TestBlockchain(unittest.TestCase):
    def setUp(self):
        self.blockchain = Blockchain()

    def test_register_grid(self):
        grid_data = {
            'name': 'Test Grid', 
            'private_key': 'YOUR_PRIVATE_KEY'  # Replace with a test private key
        }
        tx_hash = self.blockchain.register_grid(grid_data)
        self.assertTrue(tx_hash)  # Check if a transaction hash is returned

    def test_update_model(self):
        model_hash = "test_model_hash"
        tx_hash = self.blockchain.update_model("YOUR_PRIVATE_KEY", model_hash) # Replace with a test private key
        self.assertTrue(tx_hash)

if __name__ == '__main__':
    unittest.main()