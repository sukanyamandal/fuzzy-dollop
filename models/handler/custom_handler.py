from ts.torch_handler.base_handler import BaseHandler
import torch
import logging

logger = logging.getLogger(__name__)

class CustomHandler(BaseHandler):
    def preprocess(self, data):
        # Convert input data to tensor
        return torch.tensor(data)

    def postprocess(self, inference_output):
        # Convert output tensor to list
        return inference_output.detach().numpy().tolist()

    def handle(self, data, context):
        preprocessed_data = self.preprocess(data)
        model_output = self.model(preprocessed_data)
        return self.postprocess(model_output)