# Week 1 — Representation Learning and Upvote Prediction

## Overview

This project explores representation learning and feature fusion techniques through the task of predicting Hacker News upvotes.

The goal was to build end-to-end machine learning pipelines capable of:
- processing large-scale text datasets,
- training word embeddings,
- learning semantic representations,
- and predicting engagement signals from textual and metadata features.

The work focused heavily on practical experimentation, exploratory analysis, and training infrastructure.

---

# Main Objectives

- Train Word2Vec embeddings from scratch
- Compare CBOW and Skip-Gram architectures
- Build preprocessing and tokenization pipelines
- Explore feature fusion architectures
- Predict Hacker News upvotes
- Experiment with embedding fine-tuning
- Build reproducible GPU-based training workflows

---

# Dataset

The experiments used large-scale Hacker News data containing:
- titles
- authors
- websites
- metadata
- engagement statistics

Additional corpora were used for pretraining embeddings.

---

# Architecture Overview

```mermaid
flowchart LR

A[Raw Text Data]
--> B[Tokenization]

B --> C[Word2Vec Training]
C --> D[Embedding Space]

D --> E[Feature Fusion]
E --> F[Prediction Model]

F --> G[Upvote Prediction]
```

---

# Implemented Components

## Text Preprocessing

- tokenization
- normalization
- stemming / lemmatization
- vocabulary construction
- batching pipelines

## Word2Vec Training

Implemented:
- Continuous Bag of Words (CBOW)
- Skip-Gram

Explored:
- context window size
- embedding dimensionality
- negative sampling
- training stability

## Feature Fusion

Experimented with:
- early fusion
- late fusion
- metadata + text fusion

Additional features included:
- author information
- source domains
- temporal signals

## Prediction Models

Built regression and ranking-style models using:
- learned embeddings
- handcrafted features
- combined feature representations

---

# Engineering Work

## Infrastructure

- GPU training
- experiment tracking
- checkpointing
- reproducible training runs

## Deployment Experiments

- Dockerized services
- FastAPI inference endpoints
- local deployment workflows

---

# Key Topics Explored

- representation learning
- embedding geometry
- semantic similarity
- feature engineering
- large-scale preprocessing
- model evaluation
- experiment management

---

# References

- Mikolov et al. — Word2Vec
- Distributed Representations of Words and Phrases
- Efficient Estimation of Word Representations in Vector Space
