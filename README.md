# Bayesian Optimization Framework for Efficient Fleet Design in Autonomous Multi-Robot Exploration

This repository contains the code associated with the paper titled **"Bayesian Optimization Framework for Efficient Fleet Design in Autonomous Multi-Robot Exploration."** The framework leverages Bayesian Optimization to optimize the design of heterogeneous multi-robot fleets for autonomous exploration tasks.

## Overview

The goal of this project is to provide a reproducible implementation of the methods proposed in the paper. The framework is built upon a combination of Bayesian Optimization (BO) and the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm to tackle the problem of efficient fleet design in autonomous multi-robot exploration.

## Repository Structure

- **`src/`**: Contains the source code implementing the Bayesian Optimization framework and the MADDPG algorithm.
- **`experiments/`**: Includes scripts to run the experiments and reproduce the results presented in the paper.
- **`docs/`**: Documentation files related to the project.
- **`results/`**: Directory where experimental results will be stored.
- **`requirements.txt`**: A list of dependencies required to run the code.


## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/davidmccl/Bayesian_Optimization_for_Fleet_Design.git
   cd Bayesian_Optimization_for_Fleet_Design

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv env
   source env/bin/activate
3. Install the required packages:
   ```bash
    pip install -r requirements.txt

## Citation

If you use this code in your research, please cite the paper:
```bash
@misc{molinaconcha2024bayesianoptimizationframeworkefficient,
      title={Bayesian Optimization Framework for Efficient Fleet Design in Autonomous Multi-Robot Exploration}, 
      author={David Molina Concha and Jiping Li and Haoran Yin and Kyeonghyeon Park and Hyun-Rok Lee and Taesik Lee and Dhruv Sirohi and Chi-Guhn Lee},
      year={2024},
      eprint={2408.11751},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2408.11751}, 
}


