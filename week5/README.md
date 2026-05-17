# Week 5 — Audio Models and Efficient Fine-Tuning

## Overview

This project explored audio as a machine learning modality and focused on speech modeling, audio classification, and efficient fine-tuning of transformer-based systems.

The work combined:
- audio preprocessing,
- spectrogram-based modeling,
- Whisper fine-tuning,
- and parameter-efficient fine-tuning techniques.

---

# Main Objectives

- Process and analyze audio data
- Build audio classification pipelines
- Fine-tune speech models
- Explore PEFT techniques
- Experiment with distributed training workflows

---

# Audio Pipeline

```mermaid
flowchart LR

A[Raw Audio]
--> B[Spectrogram / Fourier Transform]

B --> C[Audio Encoder]

C --> D[Transformer Blocks]

D --> E[Classification / Transcription]
```

---

# Implemented Components

## Audio Processing

Worked with:
- Fourier transforms
- spectrogram generation
- mel-frequency representations
- audio normalization

## Whisper Fine-Tuning

Explored:
- speech-to-text pipelines
- decoder conditioning
- sequence generation
- transcription evaluation

## PEFT Techniques

Studied and experimented with:
- LoRA
- adapters
- prefix tuning
- mixed precision training

---

# Topics Explored

- audio embeddings
- transformer-based speech models
- parameter-efficient fine-tuning
- distributed training
- memory optimization

---

# Engineering Work

- GPU acceleration
- mixed precision
- training optimization
- experiment tracking
- reproducible workflows

---

# References

- Whisper
- LoRA
- PEFT
- Audio Spectrogram Transformer
