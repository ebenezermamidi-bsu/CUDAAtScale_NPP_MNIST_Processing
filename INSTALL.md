## System Prerequisites

This project was developed and tested on the following setup:

### Hardware
- NVIDIA GPU (Tested on RTX 4070 Laptop GPU)

### Operating System
- Windows 11 with WSL2 (Ubuntu 22.04)

### Software Requirements

#### NVIDIA Drivers
Install latest NVIDIA drivers (Windows side)

Verify:
```
nvidia-smi
```

#### CUDA Toolkit (inside WSL)
Install CUDA Toolkit for Linux (inside Ubuntu)

Verify:
```
nvcc --version
```

#### Build Tools
```
sudo apt update
sudo apt install -y build-essential make
```

#### Required Libraries
```
sudo apt install -y zlib1g-dev
```

---

## Quick Setup Script

You can run the following commands to set up everything:

```
# Update system
sudo apt update

# Install build tools
sudo apt install -y build-essential make

# Install zlib
sudo apt install -y zlib1g-dev

# Set CUDA paths (if not already set)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
