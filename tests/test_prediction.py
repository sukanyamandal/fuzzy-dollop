import unittest
from utils.prediction import Prediction
import torch

class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.prediction = Prediction()

    def test_get_prediction(self):
        # Load a sample model (you might need to adapt this based on your model architecture)
        model = torch.nn.Linear(3, 1)  
        torch.save(model.state_dict(), "test_model.pth")

        # Simulate input data
        input_data = {'input': [0.5, 0.2, 0.1]}
        result = self.prediction.get_prediction(input_data)

        # Assertions (you might need to adjust these based on your expected output)
        self.assertIn('prediction', result)  
        self.assertIsInstance(result['prediction'], float)

if __name__ == '__main__':
    unittest.main()