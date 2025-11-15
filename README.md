# Instruction Fine-Tuning with GPT-2

This repository demonstrates **Instruction Fine-Tuning of GPT-2 models from scratch**. It includes utilities for data preprocessing, model modification adding LORA module, training and infernce**
---

## Overview

- Preprocess text data for instruction fine-tuning.
- Download and modify GPT-2 models, including adding a LORA adabtors
- Train GPT-2 models on custom instruction datasets.
- Generate text using the fine-tuned model.
- we choose text to sql task

---


## Project Structure
InstructionFineTuning/
│
├── src/ # Source code
│ ├── init.py # Init file for the src package
│ ├── config.py # Hyperparameters and configuration
│ ├── create_config_file.py # Dynamic config file generator
│ ├── data_utils.py # Text data preprocessing utilities
│ ├── GPT2.py # GPT-2 model wrapper
│ ├── GPT2Modification.py # GPT-2 classifier head modifications
│ ├── inference.py # Prediction logic
│ ├── lorautils.py # LoRA utilities (if used for fine-tuning)
│ ├── model_download.py # Model downloading utilities
│ └── train_evaluate.py # Training and evaluation functions
└── requirements.txt # Python dependencies


---

## Features

- **Dynamic configuration** via `config.py`.
- **GPT-2 modifications** for instruction-following tasks.
- **LoRA support** for efficient fine-tuning.
- **Integrated training and evaluation** pipeline.
- **Text generation** for testing the fine-tuned model.

