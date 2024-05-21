# Long Short-Term Memory Neural Networks for Multi-Aspect Multi-Sentiment Analysis with Attention

An empirical investigation into Long Short-Term Memory neural network architectures with attention mechanisms evaluated on the challenge **Multi-Aspect Multi-Sentiment (MAMS)** dataset ([Jiang et al., EMNLP 2019](https://aclanthology.org/D19-1654/)).

Unlike standard Aspect-Based Sentiment Analysis (ABSA) datasets where sentences typically have a single sentiment (ie., positive, negative or neutral), MAMS data contain multiple aspects with different sentiments within the same sentence. This makes the sentiment analysis task much more challenging.

---

## Project Overview

This project systematically tests the impact of aspect integration, attention mechanisms, and network complexity on sentiment classification. Importantly, the aim of the project is not to determine the best possible model. Rather, it is to determine the effect of attention on simple recurrent models. The following models are evaluated:

- **Model 0 (Null Baseline)**: Aspect-unaware bidirectional LSTM baseline.
- **Model 1 (Aspect-Aware BiLSTM)**: Bidirectional LSTM incorporating aspect vector representations.
- **Model 2 (Simple Attention BiLSTM)**: Aspect-aware BiLSTM with attention scoring over token hidden states.
- **Model 3 (Aspect Attention BiLSTM)**: Aspect-guided attention mechanism where aspect embeddings modulate token attention weights.

### Key Findings

- **Aspect awareness is essential**: Aspect-aware models substantially outperform aspect-unaware baselines, demonstrating that aspect conditioning is required to disambiguate multi-sentiment contexts.
- **Attention improves classification**: Integrating a simple attention mechanism achieves the strongest overall performance.
- **Complexity vs. Regularization trade-offs**: Ablation studies show that increasing recurrence depth or embedding dimensionality, as well as applying heavy dropout, penalises generalization.

---

## Repo Structure

- **LSTM-MAMS.ipynb**: End-to-end PyTorch notebook (preprocessing, training, ablations, attention visualization).
- **Multi-Aspect Multi-Sentiment Analysis with Attention.pdf**: Final academic report.
- **data**: MAMS train, validation, and test splits (JSON format).
