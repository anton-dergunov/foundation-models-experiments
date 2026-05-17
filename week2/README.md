# Week 2 — Neural Search and Information Retrieval

## Overview

This project focuses on modern neural information retrieval systems and ranking architectures.

The goal was to move from classical retrieval approaches toward semantic search systems based on dense embeddings and neural ranking models.

The work included:
- retrieval pipelines,
- dual-encoder architectures,
- cross-encoders,
- vector search,
- and retrieval training objectives.

---

# Main Objectives

- Build semantic retrieval systems
- Implement dual-encoder architectures
- Experiment with cross-encoders
- Train retrieval models with triplet loss
- Build embedding-based search pipelines
- Explore ANN/vector search workflows

---

# Retrieval Pipeline

```mermaid
flowchart LR

A[User Query]
--> B[Query Encoder]

C[Documents]
--> D[Document Encoder]

B --> E[Vector Space]
D --> E

E --> F[Similarity Search]
F --> G[Top-K Results]
```

---

# Topics Covered

## Classical Retrieval

- TF-IDF
- inverted indexes
- BM25 intuition
- PageRank concepts

## Neural Retrieval

Implemented:
- dual-encoder retrieval
- cosine similarity ranking
- triplet loss optimization

Explored:
- embedding quality
- retrieval evaluation
- hard negatives
- semantic similarity

## Cross-Encoders

Studied higher-accuracy ranking approaches based on:
- token-level interactions
- joint query-document encoding
- transformer ranking architectures

---

# Datasets

## MS MARCO

Used for:
- query-document matching
- positive/negative retrieval training
- retrieval evaluation

---

# Engineering Components

- PyTorch training loops
- embedding generation pipelines
- vector indexing
- experiment tracking
- retrieval evaluation tooling

---

# Search System Stages

1. Offline training
2. Document embedding generation
3. Vector indexing
4. Online retrieval
5. Re-ranking

---

# Key Learning Areas

- semantic retrieval
- ranking systems
- embedding spaces
- ANN search
- contrastive learning
- retrieval evaluation

---

# References

- MS MARCO
- ColBERT
- Dense Passage Retrieval
- Approximate Nearest Neighbor Search
