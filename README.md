# CMPSCI 589 Final Project: ML Algorithm Evaluation

**Group Members:** Daniel Aziev-Smalovschi, Shreyas Donti, Gabriel Lojo

## Overview
This repository contains our implementations of several machine learning algorithms, including a Neural Network, Decision Tree, kNN, and Random Forest. We evaluate these models across four datasets using Stratified 10-Fold Cross-Validation.

**Datasets Evaluated:**
* Hand-Written Digits (Image Classification)
* Oxford Parkinson's Disease Detection
* Rice Grains Classification
* Credit Approval
* Fashion-MNIST (Image Classification)

## Repository Structure
* `/datasets/`: Contains the raw CSV data files.
* `/models/`: Our algorithm implementations.
* `/utils/`: Shared universal data parsing pipeline (`data_loader.py`) and evaluation loop (`evaluation.py`).
* `/experiments/`: Scripts used to run hyperparameter sweeps and generate learning curves.
* `/latex_source/`: Project report and generated plot figures.

## Setup Instructions
1. Clone this repository to your local machine.
2. Install the required dependencies:

   pip install -r requirements.txt

## Running Experiments

1. Run the command below in the terminal, replacing "run_example" with the name of the experiment file you'd like to run.
2. Make sure you do not add .py to the end as the command recognizes the file without it, example below
   ```bash
   python -m experiments.run_digits
