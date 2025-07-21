# Sequence-Based Prediction of Protein Secondary Structure
Summer Siege 2025  
ML Club  

## Problem Statement:
Develop a high-accuracy machine learning model
to predict the three-state secondary structure (α-
helix [H], β-sheet [E], or coil [C]) of individual
amino acids in a protein using only its raw
sequence data, without requiring evolutionary or
homologous sequence information. The model
should address key challenges in computational
biology while minimizing reliance on domain specific
biological knowledge.  

---

## Repository Structure

- `train.py` - Main training script for the models
- `eval.py` - Evaluation script for trained models
- `tune.py` - Hyperparameter tuning utilities
- `models/` - Model architectures (BiLSTM implementation)
- `datasets/` - Data loading and preprocessing modules
- `data/` - Raw protein sequence datasets
- `checkpoints/` - Saved model weights and training states
- `notebooks/` - Jupyter notebooks for EDA and analysis
- `results/` - Training logs and evaluation results
- `config.yaml` - Configuration file for model parameters

## Dataset Sources
- **CB513 dataset** – standard benchmark for protein secondary structure prediction. Downloaded from [Hugging Face](http://huggingface.co/datasets/proteinea/secondary_structure_prediction/tree/main)
- **Preprocessing:** Windowed sequence inputs; ESM (Evolutionary Scale Modelling) embeddings extracted for each residue.  
- No evolutionary profiles (PSSMs/MSAs) used.

## Architecture Overview

The project implements a **Bidirectional LSTM (BiLSTM)** architecture for protein secondary structure prediction:

- **Input**: Raw amino acid sequences (21 amino acids + padding)
- **Embedding Layer**: Learnable embeddings or pre-trained ESM embeddings (320-dim)
- **BiLSTM Layers**: Bidirectional LSTM to capture sequence dependencies
- **Output Layer**: Linear classifier for 3-class prediction (H, E, C)
- **Training**: Cross-entropy loss with Adam optimizer

The model supports both traditional learned embeddings and pre-trained ESM protein embeddings from [Facebook's ESM2 model](https://github.com/facebookresearch/esm).

## Results Achieved

The following table shows the performance comparison between the traditional BiLSTM model and the ESM-based BiLSTM model on the test dataset:

| Model | Q3 Accuracy | Weighted F1 Score |
|-------|-------------|-------------------|
| BiLSTM | 94.72% | 94.71% |
| ESM-BiLSTM | 89.55% | 89.53% |

*Q3 Accuracy represents the overall three-state secondary structure prediction accuracy (H, E, C)*

## Future Work

1. **Grid-search based hyperparameter optimization** with a much larger search space to choose the best of the best hyperparameters.
2. **Building a completely new transformer model** for improved sequence modeling.
3. **Multi-task learning** to jointly predict secondary structure along with other protein properties like solvent accessibility.

---
  
Pranjal Goyal  
Computer Science and Engineering  
BTech 2024  
Indian Institute of Technology, Gandhinagar
