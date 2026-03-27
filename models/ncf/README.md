## NCF: Neural Collaborative Filtering

An educational implementation of **Neural Collaborative Filtering (NeuMF)** for recommendation in PyTorch, adapted for ranking evaluation.

This project aims to capture the core idea of NCF faithfully rather than reproduce the original paper's codebase, preprocessing pipeline, or benchmark numbers exactly. The goal is understanding the model and implementing it in a way that is clear, readable, and true to the paper's core intuition.

### What is NCF?

NCF replaces the inner product used in traditional matrix factorization with a learned neural architecture. The key insight is that a fixed dot product limits the expressiveness of user-item interactions, and a neural network can learn a more complex interaction function from data.

The paper proposes three models of increasing complexity:

- **GMF** (Generalized Matrix Factorization): Element-wise product of user and item embeddings, generalizing standard matrix factorization by allowing a learned (rather than uniform) weighting of latent dimensions.

- **MLP** (Multi-Layer Perceptron): Concatenates user and item embeddings and passes them through a deep neural network, learning a nonlinear interaction function.

- **NeuMF** (Neural Matrix Factorization): Fuses GMF and MLP by concatenating their outputs and feeding them into a final prediction layer. This combines the linear interaction modeling of GMF with the nonlinear capacity of MLP.

### The Core Intuition

Traditional collaborative filtering compresses user-item interactions into a dot product:

`score(u, i) = user_embedding · item_embedding`

This assumes that user-item compatibility can be captured by a single linear operation. NCF generalizes this by learning the interaction function:

`score(u, i) = f(user_embedding, item_embedding)`

where `f` is a neural network. The GMF branch preserves the interpretability of element-wise interactions, while the MLP branch adds capacity to model complex, nonlinear patterns. NeuMF combines both to get the best of both worlds.

### Key Difference from Sequential Models

Like Wide & Deep, NCF does not model the order of user interactions. It treats recommendation as a pointwise scoring problem: given a (user, item) pair, predict whether the user would interact with the item. This makes it a strong baseline that tests whether sequential modeling actually helps beyond learning user-item compatibility.

Unlike Wide & Deep, NCF does not use any hand-crafted features. It operates purely on user and item IDs, relying entirely on learned embeddings and the neural interaction function.

### Evaluation

The model is evaluated with the same ranking setup as BERT4Rec, SASRec, and Wide & Deep. For each user, the model ranks:

- 1 positive item - the last movie the user interacted with (held out)

- 100 negative items - randomly sampled movies the user did not interact with

Results:

- HR@1: 0.1538 - In 15.38% of cases, the model ranks the correct movie at position 1.

- HR@5: 0.4483 - In 44.83% of cases, the correct movie appears within the top 5 predictions.

- NDCG@5: 0.3049 - The model not only places the correct item in the top 5, but often ranks it closer to the top positions.

- HR@10: 0.6238 - In 62.38% of cases, the correct movie appears within the top 10 predictions.

- NDCG@10: 0.3615 - Within the top 10, the correct item is typically ranked high rather than near the bottom.

- MRR: 0.2990 - On average, the correct item is ranked around 3rd place.

### Training

To train and evaluate the model, run the following from the project root:

```
python -m models.ncf.train
```
