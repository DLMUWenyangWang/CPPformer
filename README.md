# CPPformer：A Dual-Stream Framework for Disentangling Local-Global Dynamics in Time Series
This repository contains the PyTorch implementation of **CPPformer**, a hybrid time-series forecasting framework designed to address heterogeneous temporal dynamics. By introducing a novel **Dual-Stream Architecture**, the model explicitly disentangles **Local Inductive Bias** (targeting short-term fluctuations) from the **Global Receptive Field** (capturing long-term dependencies). These orthogonal features are subsequently synthesized via an **Adaptive Gate Fusion** mechanism to ensure robust predictive performance.

# Model Architecture
The model consists of two parallel branches:
1.  **Local Perception Stream (Micro-Dynamics)**: Captures local temporal patterns and short-term trends.
2.  **Global Association Stream (Macro-Dynamics)**: Captures global dependencies and long-range interactions using Self-Attention.
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
```
# Contact
For any questions, please open an issue or contact the author.
