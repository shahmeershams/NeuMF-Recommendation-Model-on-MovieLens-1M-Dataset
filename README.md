# NeuMF Recommendation Model on MovieLens 1M Dataset

This repository contains a PyTorch implementation of the Neural Matrix Factorization (NeuMF) model for recommendation systems. The model is trained and evaluated on the MovieLens 1M dataset for implicit feedback (binary interaction) prediction.

---

## Dataset

- **MovieLens 1M** dataset from [GroupLens](https://grouplens.org/datasets/movielens/1m/).
- Dataset contains 1 million ratings from 6000+ users on 4000+ movies.
- We use the `ratings.dat` file and convert ratings to binary interactions (`rating >= 4` as positive interaction).

---

## Features

- NeuMF model combining Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) for user-item interaction.
- Training with binary cross-entropy loss.
- Evaluation metrics include RMSE and NDCG@10.
- Supports training on full or sampled subsets of the data.
- Optionally supports batch training using PyTorch DataLoader.

---

## Setup and Usage

### 1. Download and preprocess dataset

```python
!wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip -q ml-1m.zip

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load and preprocess
df = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python',
                 names=['user_id', 'movie_id', 'rating', 'timestamp'])
df.drop('timestamp', axis=1, inplace=True)

# Encode IDs to consecutive integers
user_enc = LabelEncoder()
item_enc = LabelEncoder()
df['user'] = user_enc.fit_transform(df['user_id'])
df['item'] = item_enc.fit_transform(df['movie_id'])

# Binary interaction (rating >= 4)
df['interaction'] = (df['rating'] >= 4).astype(int)

# Train-test split
train_df, test_df = train_test_split(df[['user', 'item', 'interaction']], test_size=0.2, random_state=42)

