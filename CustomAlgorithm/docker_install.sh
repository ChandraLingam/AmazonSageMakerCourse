#!/bin/bash

# Check if /etc/os-release file exists
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [ "$ID" != "ubuntu" ]; then
        echo "This is not JupyterLab Environment. Exiting the script."
        exit 1
    fi
else
    echo "/etc/os-release file not found. Exiting the script."
    exit 1
fi

if command -v docker >/dev/null 2>&1; then
    echo "Docker is already installed. Skipping installation."
    exit 1
fi


# Update the package list and install necessary packages
sudo apt-get -y update
sudo apt-get -y install ca-certificates curl gnupg

# Install Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add Docker repository to the APT sources
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update the package list again
sudo apt-get -y update

# Install specific version of Docker Client and Docker Compose
VERSION_STRING="5:20.10.24~3-0~ubuntu-jammy"
sudo apt-get install docker-ce-cli=$VERSION_STRING docker-compose-plugin -y

# Validate Docker installation
docker version
