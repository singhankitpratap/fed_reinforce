# Federated REINFORCE

## Overview
Federated REINFORCE adapts the classic REINFORCE policy gradient algorithm for distributed, privacy-preserving reinforcement learning using federated learning techniques. Robust aggregation methods, including the geometric median, are implemented to ensure security and efficiency.

## Requirements
Install the following dependencies before running the code:
- `gym`: For creating and comparing reinforcement learning environments.
- `numpy`: Numerical operations.
- `torch` (PyTorch): Machine learning library.
- `seaborn`: For data visualization.
- `geom_median`: Install using `!pip install geom_median` for robust aggregation.

## Key Features
- Distributed training using federated learning.
- Robust aggregation via geometric median.
- Detailed visualization and statistical insights.
