#!/bin/bash

# Install Node.js
curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Truffle
sudo npm install -g truffle

# Install Ganache
sudo npm install -g ganache-cli