# S²-BiHT

This repository provides the official implementation and trained model parameters for the paper:

**Spatial-Spectral Bi-Hemispheric Transformer for Asymmetry-Preserving EEG Emotion Recognition**

S²-BiHT is a neuroscience prior-guided transformer framework for cross-subject EEG emotion recognition. It explicitly integrates EEG spatial topology, spectral information, hemispheric organization, and asymmetry-preserving domain adaptation into a unified deep learning model.

---

## Brief Introduction

Electroencephalography (EEG)-based emotion recognition is an important technique for affective brain-computer interfaces. However, cross-subject EEG emotion recognition remains challenging due to three key issues:

- complex spatial-spectral dependencies across EEG electrodes and frequency bands;
- hemispheric asymmetry related to affective neural processing;
- substantial inter-subject variability in EEG signals.

Most existing deep learning models rely on global feature learning or dense attention mechanisms. Although these approaches can capture long-range dependencies, they may also introduce redundant spatial interactions and weaken lateralized emotional information. In addition, conventional domain adaptation strategies often align whole-brain features globally, which may suppress emotion-related hemispheric differences.

To address these problems, this project proposes **S²-BiHT**, a Spatial-Spectral Bi-Hemispheric Transformer for EEG emotion recognition. The model consists of three core components:

- **Spatial-Spectral Representation (S²R)**  
  Maps multi-channel EEG differential entropy features into a topology-aware 9 × 9 spatial grid and constructs region-level spatial-spectral tokens.

- **Bi-Hemispheric Routing (BiHR)**  
  Separates EEG representations into left hemispheric, midline-anchor, and right hemispheric streams. It uses distance-constrained attention and midline-anchor-based routing to regulate cross-hemispheric information exchange.

- **Asymmetry-Preserving Domain Adaptation (APDA)**  
  Constructs a hemispheric-difference representation and performs stream-wise adversarial alignment to reduce subject-specific distribution shifts while preserving discriminative hemispheric asymmetry.

The overall goal of S²-BiHT is to improve cross-subject EEG emotion recognition while providing a more interpretable modeling framework based on neurophysiological priors.

---

## Framework
<img width="1449" height="551" alt="dfc75eaa4f041652bfa46a3f1a129e35" src="https://github.com/user-attachments/assets/5549eef6-bed0-4cca-a115-b4ca2910a927" />

---

## Code Structure Overview

The main files and directories included in this repository are:

* **`seed_S2_BiHT.py`**: Contains the model definition and implementation of the Spatial-Spectral Bi-Hemispheric Transformer (S²-BiHT) for EEG emotion recognition on the SEED dataset. It includes the core modules of spatial-spectral representation, bi-hemispheric routing, asymmetry-preserving domain adaptation, and emotion classification.​
* **`seed_model`**: A directory for storing trained model parameters or pre-trained configurations related to the S²-BiHT model on the SEED dataset.

---

## Data
* **`seed`**: https://bcmi.sjtu.edu.cn/home/seed/seed.html
* **`seed-iv`**: https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html
* **`dreamer`**: https://zenodo.org/records/546113
