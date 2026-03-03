# CPPformer：Time Series Forecasting with Adaptive Gate Fusion
This repository contains the PyTorch implementation of **CPPformer**, a hybrid time-series forecasting model that combines **LSTM** (for local/short-term dependencies) and **Transformer** (for global/long-term dependencies) using an **Adaptive Gate Fusion** mechanism.

# Model Architecture
The model consists of two parallel branches:
1.  **LSTM Branch**: Captures local temporal patterns and short-term trends.
2.  **Transformer Branch**: Captures global dependencies and long-range interactions using Self-Attention.
3.  **Adaptive Gate Fusion**: A learnable gating mechanism that dynamically weights the contributions of the LSTM and Transformer outputs based on the input features.

# Requirements
To install all necessary dependencies, please run the following command:
```bash
pip install -r requirements.txt
```

# Data 
The dataset provided in this repository (`dataset`) contains raw data. 

# Directory Structure
```text
CPPformer/
├── dataset/            
├── run.py              
├── requirements.txt    
└── README.md           
