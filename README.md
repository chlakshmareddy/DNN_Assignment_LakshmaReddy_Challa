# Multimodal Story Continuation using Cross-Modal Attention
Author: Lakshma Reddy Challa

# Project Structure

.
├── data/                  # Dataset loaders and preprocessing
├── models/                # Model architectures and attention modules
├── training/              # Training and evaluation scripts
├── results/
│   ├── figures/           # Plots and qualitative results
│   └── logs/              # Training logs
├── utils/                 # Helper functions
├── README.md
└── requirements.txt

Introduction and Problem Statement

This repository contains the final assessment for the Deep Neural Networks and Learning Systems (DNNLS) module. The objective of this project is to extend a provided baseline multimodal storytelling model by implementing an architectural modification and evaluating its impact on multimodal alignment and narrative prediction quality.

The task focuses on visual storytelling, where a model must interpret a sequence of image–text pairs and generate the next image description in the sequence. This requires temporal reasoning, cross-modal understanding, and sequence modelling.

The project uses the Image-Based Storytelling Dataset for training and evaluation.

Problem Definition

Given a baseline multimodal encoder–decoder architecture, the project aims to:

Baseline Approach: Encode images and XML text independently, then fuse their embeddings using simple concatenation.

Innovation (Proposed Change): Replace concatenation with a Cross-Modal Attention Fusion Layer that enables text embeddings to attend over visual feature maps.

Performance is evaluated using quantitative caption-generation metrics and qualitative inspection of outputs.

Methods
Model Architecture Overview

The model consists of the following key components:

Image Encoder: ResNet-18 CNN

Text Encoder: LSTM network

Fusion Block:

Baseline: concatenation

Modified: cross-modal attention

Sequence Model: GRU for temporal reasoning

Decoders:

Text decoder for caption generation

Lightweight image decoder for predicting the next frame representation

A high-level diagram of the architecture is located at:

results/figures/architecture_diagram.png

Cross-Modal Attention Fusion

The proposed fusion mechanism extends standard feature concatenation by introducing scaled dot-product attention between modalities. In this design:

Text embeddings act as queries

Image embeddings act as keys and values

The attention mechanism allows the model to prioritise visual regions that are semantically relevant to the provided text signal.

This modification aims to improve multimodal alignment and enhance narrative coherence in predicted captions.

Results
Quantitative Evaluation

Model performance was assessed using BLEU-4 and ROUGE-L:

Model	BLEU-4	ROUGE-L
Baseline (Concat)	1.00e-09	0.00000
Cross-Modal Attention	8.064406e-06	0.27866

Relevant visualisations are located in:

results/figures/performance_comparison.png

Qualitative Analysis

Example predicted captions and comparisons are provided in:

results/figures/caption_comparison.png

These illustrate differences in narrative fluency and alignment quality.

Conclusions

The Cross-Modal Attention Fusion Layer demonstrated improved text alignment performance compared to the baseline concatenation approach. Although BLEU-4 values remain extremely small due to the short XML captions, ROUGE-L scores indicate that attention improves structural similarity to reference descriptions.

The findings suggest that attention-based fusion can enhance multimodal alignment, though performance remains constrained by the limited semantic richness of the dataset annotations.

Future Work

Evaluate on richer narrative datasets (e.g., VIST)

Introduce Transformer-based text encoders

Explore contrastive multimodal training objectives

Investigate higher-quality visual decoders

Conduct human-judgement narrative coherence scoring.
