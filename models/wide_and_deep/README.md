## Wide & Deep: Wide & Deep Learning for Recommender Systems

An educational implementation of **Wide & Deep** for recommendation in PyTorch, adapted for ranking evaluation comparable to BERT4Rec and SASRec.

This project aims to capture the core idea of Wide & Deep faithfully rather than reproduce the original paper's production system. The goal is understanding the model and implementing it in a way that is clear, readable, and true to the paper's core intuition.

### What is Wide & Deep?

Wide & Deep combines two complementary components:

- **Wide** (memorization): A linear model on hand-crafted cross-product features that memorizes specific feature combinations from the data.

- **Deep** (generalization): An MLP over dense embeddings that generalizes to unseen feature combinations through learned representations.

By jointly training both components, the model benefits from the memorization strength of the wide part and the generalization ability of the deep part.

### Wide Features

The wide component receives a 54-dimensional feature vector per (user, item) pair:

- User genre profile (18-dim): average genre vector over the user's training history
- Item genres (18-dim): binary genre vector of the candidate movie
- Cross-product (18-dim): element-wise product of user profile and item genres

This captures memorization patterns like "users who watch a lot of action movies tend to watch this action movie".

### Deep Features

The deep component learns dense embeddings for user IDs and movie IDs, concatenates them, and passes them through an MLP. This allows the model to generalize beyond explicit feature interactions.

### Key Difference from Sequential Models

Unlike BERT4Rec and SASRec, Wide & Deep does not model the order of user interactions. It treats recommendation as a pointwise scoring problem: given a (user, item) pair, predict whether the user would interact with the item. This makes it a strong baseline that tests whether sequential modeling actually helps beyond simple user-item compatibility.

### Evaluation

The model is evaluated with the same ranking setup as BERT4Rec and SASRec. For each user, the model ranks:

- 1 positive item - the last movie the user interacted with (held out)

- 100 negative items - randomly sampled movies the user did not interact with

Results:

- HR@1: 0.1614 - In 16.14% of cases, the model ranks the correct next movie at position 1 (best prediction).

- HR@5: 0.4828 - In 48.28% of cases, the correct movie appears within the top 5 predictions.

- NDCG@5: 0.3270 - The model not only places the correct item in the top 5, but often ranks it closer to the top positions.

- HR@10: 0.6430 - In 64.30% of cases, the correct movie appears within the top 10 predictions.

- NDCG@10: 0.3790 - Within the top 10, the correct item is typically ranked high rather than near the bottom.

- MRR: 0.3139 - On average, the correct item is ranked around 3rd place.

### Training

To train and evaluate the model, run the following from the project root:

```
python -m models.wide_and_deep.train
```
