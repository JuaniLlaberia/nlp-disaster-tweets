# NLP Disaster Tweets — Kaggle Competition

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)
![Metric](https://img.shields.io/badge/Metric-F1--Score-green)
![Rank](https://img.shields.io/badge/Rank-56th-orange)

Fine-tuning transformer-based models for binary classification of disaster-related tweets.  
This project focuses on maximizing **F1-score** through careful preprocessing, threshold tuning, and hyperparameter optimization.

---

## Introduction

The goal of this competition is to classify whether a tweet refers to a real disaster or not.  
This is a classic **imbalanced NLP classification problem**, where optimizing **F1-score** is more important than accuracy.

Key challenges:
- Short and noisy text (tweets)
- Ambiguity and sarcasm
- Class imbalance
- Sensitivity to threshold selection

My approach focused on:
- Strong transformer baselines
- Careful validation strategy
- Threshold optimization
- Hyperparameter tuning

Final result: **🏅 Rank 56**  
(*Note: Top ~10 submissions likely affected by data leakage due to perfect scores*)

---

## Experiments

### Baseline

- **TF-IDF + Logistic Regression**
- Provided a solid starting point but limited by feature representation

| Model | Val F1 | Test F1 |
|------|--------|--------|
| TF-IDF + LR | ~0.77 | ~0.74 |

---

### Transformer Models

#### BERTweet / Transformer Fine-Tuning

- Pretrained transformer models fine-tuned on the dataset
- Focus on:
  - Learning rate
  - Batch size
  - Weight decay
  - Number of epochs
  - Layer freezing strategy

Key techniques:
- Partial layer unfreezing
- Regularization via weight decay
- Careful training stability tuning

---

### Threshold Optimization ⚡

Instead of using default `0.5`, optimized threshold on validation set:

- Significant boost in F1-score
- Critical for imbalanced classification

---

### Hyperparameter Optimization

- Performed multiple trials (Optuna-style search)
- Tuned:
  - Learning rate
  - Scheduler
  - Warmup steps
  - Dropout

---

### Seed Sensitivity

- Observed variation of **±0.5% F1**
- Final performance averaged across stable runs

---

### Final Model

- Transformer-based model (fine-tuned)
- Optimized threshold
- Carefully tuned hyperparameters

Result:
- **Validation F1:** ~0.83+
- **Leaderboard Rank:** **56**

---

## Conclusion

This project highlights that in NLP competitions:

- **Strong baselines + small optimizations → big gains**
- **Threshold tuning is critical** for F1-based tasks
- **Validation strategy matters as much as the model**
- Transformer models dominate, but require careful tuning

Key takeaways:
- Don’t rely on default thresholds
- Hyperparameter tuning is essential even for pretrained models
- Seed variability should not be ignored
- Simplicity + consistency often beats complexity

---

Future improvements:
- Ensembling multiple transformer models
- Better data augmentation for tweets

---
