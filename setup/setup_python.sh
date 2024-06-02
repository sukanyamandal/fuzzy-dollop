#!/bin/bash

# Install Python 3.8+
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip

# Install Python dependencies
pip install -r requirements.txt