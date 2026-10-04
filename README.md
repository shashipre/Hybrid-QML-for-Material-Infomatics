# Hybrid Quantum ML for Exfoliation Energy Prediction

This project implements a hybrid quantum-classical machine learning model for predicting the exfoliation energy of 2D materials from their crystal structures using the MatBench dataset.

## Architecture
Crystal Structure → Crystal Graph → GNN → QNN (VQC) → MLP → Energy

## Quick Start

1. **Install dependencies:**
   ```bash
   conda create -n qmat python=3.10 -y
   conda activate qmat
   pip install -r requirements.txt
   ```

2. **Run Hybrid Quantum GNN (Full 5-fold CV):**
   ```bash
   python experiments/run_training.py --config configs/default.yaml
   ```

3. **Run Single Fold (Fast Debugging):**
   ```bash
   python experiments/run_training.py --config configs/default.yaml --fold 0
   ```

4. **Run Classical Baseline Comparison:**
   ```bash
   python experiments/run_baseline.py --config configs/default.yaml --fold 0
   ```

5. **Run Test Suite:**
   ```bash
   pytest tests/ -v
   ```

## Tech Stack
- Python 3.10+
- PyTorch & PyTorch Geometric (`torch_geometric`)
- PennyLane (`pennylane`)
- Pymatgen & MatBench (`pymatgen`, `matbench`)
- Scikit-Learn & NumPy
