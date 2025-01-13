#!/bin/bash

# Update the package index
echo "Updating package index..."
sudo apt update

# Install NVIDIA driver
echo "Installing NVIDIA driver..."
sudo apt install -y nvidia-driver-535

# Reboot the system to activate the driver
echo "Rebooting the system to apply NVIDIA driver changes..."
sudo reboot

# After reboot, log back in and run the remaining steps:
# Install CUDA repository and toolkit
echo "Adding CUDA repository and installing toolkit..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2204-11-8-local/7fa2af80.pub
sudo apt update
sudo apt install -y cuda

# Install cuDNN library
echo "Installing cuDNN library..."
wget https://developer.download.nvidia.com/compute/cuda/cudnn/secure/8.4.1/local_installers/cudnn-11.8-linux-x64-v8.4.1.50.tgz
tar -xzvf cudnn-11.8-linux-x64-v8.4.1.50.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# Add CUDA to PATH and LD_LIBRARY_PATH
echo "Configuring environment variables..."
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify NVIDIA driver
echo "Verifying NVIDIA driver installation..."
nvidia-smi

# Verify CUDA installation
echo "Verifying CUDA installation..."
nvcc --version

echo "Setup complete! TensorFlow GPU environment should now be ready."

