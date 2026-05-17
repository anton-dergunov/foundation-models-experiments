# Foundation Models Experiments

A collection of deep learning implementations, retrieval systems, transformer architectures, multimodal experiments, and LLM alignment projects developed during an intensive independent ML engineering residency and research-focused sabbatical.

This repository focuses on understanding modern machine learning systems from first principles through practical implementation, experimentation, and end-to-end training workflows.

The work spans:
- representation learning
- information retrieval
- transformer architectures
- multimodal systems
- speech/audio models
- fine-tuning large language models
- retrieval-augmented generation (RAG)
- RLHF and preference optimization

The emphasis is on:
- implementing models from scratch
- understanding training dynamics
- reproducible experimentation
- GPU-based training workflows
- practical ML engineering

---

## Background

Part of this work was developed while participating in advanced deep learning workshops at the [Machine Learning Institute](https://ml.institute/learn).

The implementations and experiments in this repository are my own and were further extended independently beyond the workshop material.

---

# Repository Structure

```text
week1/   Representation learning and Hacker News prediction
week2/   Information retrieval and neural search
week3/   Transformers and vision transformers
week4/   Multimodal learning and image captioning
week5/   Audio models and efficient fine-tuning
week6/   RLHF, PPO, preference optimization, and MoE
```

---

# Technical Stack

## Core Frameworks
- PyTorch
- NumPy
- Pandas
- Hugging Face Transformers
- Weights & Biases

## Infrastructure
- Docker
- FastAPI
- Remote GPU training
- Mixed precision training
- Experiment tracking

## Topics Covered
- Word2Vec
- Retrieval systems
- Dual encoders
- Cross encoders
- Triplet loss
- Transformers
- Vision Transformers (ViT)
- Image captioning
- CLIP-style multimodal systems
- Whisper fine-tuning
- LoRA / PEFT
- RLHF
- PPO
- Mixture of Experts (MoE)
- RAG systems

---

# Weekly Projects

## Week 1 - Representation Learning and Upvote Prediction

Implemented representation learning pipelines and predictive models for Hacker News upvote prediction using:
- Word2Vec
- CBOW / Skip-Gram
- feature fusion architectures
- embedding fine-tuning
- regression models

Key areas:
- large-scale text preprocessing
- exploratory data analysis
- embedding training
- multimodal feature fusion
- experiment management

→ See [week1/README.md](week1/README.md)

---

## Week 2 - Neural Information Retrieval

Implemented retrieval and ranking systems using:
- TF-IDF
- dual-encoder architectures
- cross-encoders
- triplet loss training
- vector search pipelines

Datasets:
- MS MARCO
- custom retrieval datasets

Topics:
- semantic search
- embedding spaces
- retrieval evaluation
- vector databases
- ranking architectures

→ See [week2/README.md](week2/README.md)

---

## Week 3 - Transformers and Vision Transformers

Implemented transformer architectures from scratch:
- multi-head attention
- encoder-decoder transformers
- positional embeddings
- Vision Transformers (ViT)

Projects included:
- image-to-text systems
- sequence generation
- custom transformer layers
- autoregressive decoding

→ See [week3/README.md](week3/README.md)

---

## Week 4 - Multimodal Learning

Built multimodal pipelines combining vision and language models.

Topics:
- CLIP-style architectures
- multimodal embeddings
- image captioning
- transfer learning
- vision-language alignment

Projects:
- encoder-decoder image captioning systems
- multimodal retrieval experiments
- synthetic dataset generation

→ See [week4/README.md](week4/README.md)

---

## Week 5 - Audio Models and Efficient Fine-Tuning

Explored audio and speech modeling using:
- CNNs for audio
- transformer-based audio architectures
- Whisper fine-tuning
- parameter-efficient fine-tuning (PEFT)

Topics:
- spectrogram processing
- Fourier transforms
- LoRA
- mixed precision training
- distributed training

→ See [week5/README.md](week5/README.md)

---

## Week 6 - RLHF, PPO, and Model Alignment

Implemented and studied modern LLM alignment techniques:
- reward modeling
- preference optimization
- PPO
- RLHF pipelines
- Mixture of Experts

Topics:
- supervised fine-tuning (SFT)
- policy optimization
- KL regularization
- sparse expert routing
- alignment training workflows

→ See [week6/README.md](week6/README.md)

---

# Goals of This Repository

This repository is intentionally implementation-focused.

The primary goals are:
- understanding modern ML systems from first principles
- reproducing important architectures
- experimenting with training strategies
- building intuition around optimization and scaling
- developing production-oriented ML engineering skills

---

# References

## Core Papers

### Attention Is All You Need
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017. Introduced the Transformer architecture based entirely on self-attention, replacing recurrent and convolutional networks for sequence transduction.

### Word2Vec
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) - Mikolov et al., 2013. Introduced the CBOW and Skip-gram architectures for learning word embeddings.
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) - Mikolov et al., 2013. Extended Word2Vec with negative sampling and phrase representations.

### CLIP
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) - Radford et al., 2021. Trained an image and text encoder jointly via contrastive learning on 400M image-text pairs, enabling strong zero-shot image classification.

### ViT
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) - Dosovitskiy et al., 2020. Applied the Transformer architecture directly to sequences of image patches, achieving state-of-the-art results on image classification with sufficient pre-training data.

### LoRA
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - Hu et al., 2021. Proposed injecting trainable low-rank matrices into frozen pretrained weights, reducing trainable parameters by up to 10,000× compared to full fine-tuning.

### PPO
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017. Introduced a family of policy gradient methods that clip the surrogate objective for stable, sample-efficient reinforcement learning.

### RLHF
- [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) - Christiano et al., 2017. Foundational RLHF paper showing RL agents can learn complex behaviors from pairwise human preference feedback.
- [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) - Stiennon et al., 2020. Applied RLHF to text summarization using a learned reward model trained on human preference data.
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) - Ouyang et al., 2022. Introduced InstructGPT, applying RLHF to align GPT-3 with human intent; smaller models were preferred over larger unaligned ones.

### Switch Transformers
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) - Fedus, Zoph & Shazeer, 2021. Simplified Mixture-of-Experts routing to a single expert per token, enabling sparse models with trillions of parameters at constant compute cost.

---

## Datasets

- **Hacker News** - Full archive of all stories, comments, Ask HNs, and polls from Y Combinator's Hacker News community (2006–present). Available as a [BigQuery public dataset](https://console.cloud.google.com/marketplace/product/y-combinator/hacker-news) (`bigquery-public-data.hacker_news`) or on [Hugging Face](https://huggingface.co/datasets/open-index/hacker-news).

- **MS MARCO** (Microsoft Machine Reading Comprehension) - Large-scale dataset of real anonymized Bing queries with human-generated answers, used for question answering, passage ranking, and document retrieval. [Official page](https://microsoft.github.io/msmarco/), [Hugging Face](https://huggingface.co/datasets/microsoft/ms_marco).

- **Flickr30k** - 31,783 images paired with five human-written captions each, widely used as a benchmark for image captioning and cross-modal retrieval. [Dataset page (Illinois)](https://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.html).

- **MNIST** - 70,000 grayscale 28×28 images of handwritten digits (0–9); the canonical benchmark for image classification and learning algorithms. [Official page (Yann LeCun)](https://yann.lecun.org/exdb/mnist/).

---

# Disclaimer

This repository is intended for educational, research, and experimentation purposes.
Some implementations prioritize clarity and experimentation over production optimization.
