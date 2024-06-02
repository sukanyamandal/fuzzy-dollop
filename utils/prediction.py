import requests

class Prediction:
    def __init__(self):
        self.url = 'http://localhost:8080/predictions/smartgridmodel'

    def get_prediction(self, input_data):
        response = requests.post(self.url, json=input_data)
        return response.json()