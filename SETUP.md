# ConvE Environment Setup Guide

This guide provides step-by-step instructions to set up the ConvE environment with all dependencies.

## Prerequisites

- Ubuntu/Linux system
- Python 3.10
- Git
- CUDA-compatible GPU (optional, but recommended)

## Setup Steps

### 1. Clone the Repository

```bash
# Clone the main ConvE repository
cd /data
git clone https://github.com/TimDettmers/ConvE.git
cd ConvE
```

### 2. Clone Required Libraries

Clone the spodernet and bashmagic libraries into the ConvE directory:

```bash
# Clone spodernet (preprocessing framework for NLP)
git clone https://github.com/TimDettmers/spodernet.git

# Clone bashmagic (bash utilities)
git clone https://github.com/TimDettmers/bashmagic.git
```

Your directory structure should now look like:
```
ConvE/
├── bashmagic/
├── spodernet/
├── main.py
├── model.py
├── requirements.txt
└── ... (other ConvE files)
```

### 3. Create Python Virtual Environment

```bash
# Create a virtual environment using Python 3.10
python3.10 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### 4. Upgrade pip

```bash
# Upgrade pip to the latest version
pip install --upgrade pip
```

### 5. Install PyTorch with CUDA Support

```bash
# Install PyTorch 2.4.1 with CUDA 12.1 support
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
```

This will also install all CUDA dependencies:
- nvidia-cudnn-cu12
- nvidia-cublas-cu12
- nvidia-cuda-runtime-cu12
- And other CUDA libraries

### 6. Install Spodernet Dependencies

```bash
# Install spodernet requirements
pip install -r spodernet/requirements.txt
```

This installs:
- pytest
- cython
- numpy
- h5py
- nltk
- enum34
- simplejson
- spacy
- scikit-learn
- future

### 7. Fix Spodernet Setup (Important!)

The spodernet setup.py uses the deprecated `sklearn` package. Fix it:

```bash
# Edit spodernet/setup.py and change 'sklearn' to 'scikit-learn'
# Line 34: Change 'sklearn' to 'scikit-learn'
```

Or use this command:
```bash
sed -i "s/'sklearn'/'scikit-learn'/g" spodernet/setup.py
```

### 8. Install Spodernet Package

```bash
# Install spodernet in editable mode
pip install -e spodernet/
```

### 9. Install Bashmagic Package

```bash
# Install bashmagic in editable mode
pip install -e bashmagic/
```

### 10. Install ConvE Requirements

```bash
# Install main ConvE requirements
pip install -r requirements.txt
```

This installs:
- scipy

### 11. Verify Installation

Test that all libraries are properly installed:

```bash
python -c "
import torch
import spodernet
import bashmagic
import numpy as np
import scipy

print('✓ PyTorch version:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ Spodernet imported successfully')
print('✓ Bashmagic imported successfully')
print('✓ NumPy version:', np.__version__)
print('✓ SciPy version:', scipy.__version__)
print('\nAll libraries installed and working correctly!')
"
```

Expected output:
```
✓ PyTorch version: 2.4.1+cu121
✓ CUDA available: True
✓ Spodernet imported successfully
✓ Bashmagic imported successfully
✓ NumPy version: 2.2.6
✓ SciPy version: 1.15.3

All libraries installed and working correctly!
```

## Quick Setup Script

Here's a complete script to automate the entire setup:

```bash
#!/bin/bash

# Navigate to data directory
cd /data

# Clone repositories
git clone https://github.com/TimDettmers/ConvE.git
cd ConvE
git clone https://github.com/TimDettmers/spodernet.git
git clone https://github.com/TimDettmers/bashmagic.git

# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# Install spodernet dependencies
pip install -r spodernet/requirements.txt

# Fix spodernet setup.py (change sklearn to scikit-learn)
sed -i "s/'sklearn'/'scikit-learn'/g" spodernet/setup.py

# Install spodernet and bashmagic
pip install -e spodernet/
pip install -e bashmagic/

# Install ConvE requirements
pip install -r requirements.txt

# Verify installation
python -c "
import torch
import spodernet
import bashmagic
print('Setup completed successfully!')
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
"
```

## Daily Usage

Every time you want to work with ConvE:

```bash
cd /data/ConvE
source .venv/bin/activate
```

To deactivate the virtual environment:
```bash
deactivate
```

## Troubleshooting

### Issue: "No space left on device"
- Clean up disk space or use a different directory
- Remove pip cache: `pip cache purge`

### Issue: Virtual environment corrupted
- Remove and recreate: `rm -rf .venv && python3.10 -m venv .venv`

### Issue: CUDA not available
- Check GPU drivers: `nvidia-smi`
- Verify CUDA installation
- Reinstall PyTorch with correct CUDA version

### Issue: Import errors
- Ensure virtual environment is activated
- Verify installation: `pip list | grep -E "torch|spodernet|bashmagic"`

## Package Versions

- Python: 3.10
- PyTorch: 2.4.1 (with CUDA 12.1)
- NumPy: 2.2.6
- SciPy: 1.15.3
- Spodernet: 0.0.1
- Bashmagic: 0.0.1

## Additional Notes

- The virtual environment (.venv) is located in the ConvE root directory
- Both spodernet and bashmagic are installed in editable mode (-e flag), so changes to their code will be reflected immediately
- The setup assumes you have CUDA 12.1 compatible drivers installed
- Total download size is approximately 2.5GB for all packages

## References

- ConvE: https://github.com/TimDettmers/ConvE
- Spodernet: https://github.com/TimDettmers/spodernet
- Bashmagic: https://github.com/TimDettmers/bashmagic
- PyTorch: https://pytorch.org/
