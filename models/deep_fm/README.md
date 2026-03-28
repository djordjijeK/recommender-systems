## DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

An educational implementation of **DeepFM** for recommendation in PyTorch, adapted for ranking evaluation.

This project aims to capture the core idea of DeepFM faithfully rather than reproduce the original paper's codebase, preprocessing pipeline, or benchmark numbers exactly. The goal is understanding the model and implementing it in a way that is clear, readable, and true to the paper's core intuition.

### What is DeepFM?

DeepFM combines a Factorization Machine (FM) component with a Deep component that share the same feature embeddings. This shared-embedding design is the paper's key contribution over Wide & Deep, which requires separate feature engineering for its wide and deep parts.

The model has two parallel branches:

- **FM** (low-order interactions): Captures first-order feature importance (bias per feature) and all pairwise second-order interactions between feature fields through latent vector dot products, computed efficiently via the identity `0.5 * (||sum V_i||^2 - sum ||V_i||^2)`.

- **Deep** (high-order interactions): An MLP that takes the *same* latent vectors as input and learns complex, nonlinear feature interaction patterns.

The final prediction is the sum of both components: `y = y_FM + y_Deep`.

### The Core Intuition

Traditional recommender systems either model explicit feature interactions (like FM) or learn implicit ones (like deep neural networks). DeepFM argues that you need both:

- FM excels at capturing pairwise interactions (e.g., "male users aged 25-34 tend to watch action movies") with strong mathematical guarantees, even with sparse data.

- The Deep component can discover higher-order patterns that pairwise interactions miss.

By sharing embeddings, both components jointly refine the same representations - the FM provides a strong linear foundation while the Deep component adds nonlinear capacity on top. No feature engineering is required.

### Key Differences from Other Models

Unlike **Wide & Deep**, DeepFM requires no hand-crafted cross-product features for the FM component - feature interactions are learned automatically from the shared embeddings.

Unlike **NCF**, which uses separate embedding tables for its GMF and MLP branches and only operates on user and movie IDs, DeepFM incorporates rich side information (demographics, genres) and forces the FM and Deep components to share the same latent vectors.

### Evaluation

The model is evaluated with the same ranking setup as BERT4Rec, SASRec, Wide & Deep, and NCF. For each user, the model ranks:

- 1 positive item - the last movie the user interacted with (held out)

- 100 negative items - randomly sampled movies the user did not interact with

Results:

- HR@1: 0.1692 - In 16.92% of cases, the model ranks the correct movie at position 1 (best prediction).

- HR@5: 0.4868 - In 48.68% of cases, the correct movie appears within the top 5 predictions.

- NDCG@5: 0.3315 - The model not only places the correct item in the top 5, but often ranks it closer to the top positions.

- HR@10: 0.6671 - In 66.71% of cases, the correct movie appears within the top 10 predictions.

- NDCG@10: 0.3898 - Within the top 10, the correct item is typically ranked high rather than near the bottom.

- MRR: 0.3208 - On average, the correct item is ranked around 3rd place.

### Training

To train and evaluate the model, run the following from the project root:

```
python -m models.deep_fm.train
```
