## Self-Attentive Sequential Recommender (SASRec)

A PyTorch implementation of SASRec - Self-Attentive Sequential Recommendation (Kang & McAuley, 2018) - applied to movie recommendation using the MovieLens dataset.

The goal is to predict which movie a user will watch next given their watch history. 
Rather than treating all past interactions equally, SASRec uses a Transformer-style self-attention mechanism to identify which items in the user's history are most relevant to the next prediction - recent items often matter more, but the model learns this adaptively.


### How the Model Works

The model is a stacked self-attention architecture that processes a user's interaction history as a sequence:

1. Embedding layer - Each item in the sequence is mapped to a learned vector. 
A positional embedding is added so the model is aware of the order of interactions.


2. Causal self-attention blocks - The sequence is passed through several self-attention blocks. 
Each block computes attention scores between all pairs of positions, but applies a causal mask so each position can only attend to itself and earlier positions (no future leakage). 
A padding mask prevents padded positions from influencing results. 
This is followed by a position-wise feed-forward network, with residual connections and layer normalisation around each sub-layer.


3. Prediction head - The output vector at the final sequence position serves as a context representation for the user. 
A dot product is taken against all item embeddings to produce a ranked list of candidate items.


### Training

- Dataset & splits: The MovieLens dataset is filtered to retain only users and movies with at least 17 interactions (applied iteratively until stable). 
The data is split chronologically per user: the most recent interaction is held out for testing, the second most recent for validation, and all earlier interactions form the training set.


- Training objective: The model is trained with binary cross-entropy loss over positive/negative pairs. 
For every position in the training sequence, the model scores the actual next item (positive) against a randomly sampled item the user has not seen (negative). 
The loss pushes positive scores up and negative scores down.


- Optimiser: Adam with a learning rate of 1e-3, trained for 30 epochs with a batch size of 128.


### Evaluation

At evaluation time, each user's most recent held-out item is ranked against 64 randomly sampled unseen items (65 candidates total). 
Two metrics are computed at cutoff k = 10:

- Hit@10 - the fraction of users for whom the ground-truth item appears in the top 10.

- NDCG@10 - normalised discounted cumulative gain, which rewards ranking the correct item higher within the top 10.


### Results

| Split | Hit@10 | NDCG@10 |
|---|---|---|
| Validation (epoch 30) | 0.5271 | 0.2948 |
| Test | 0.5198 | 0.2923 |


## Reference

Kang, W.-C., & McAuley, J. (2018). *Self-Attentive Sequential Recommendation*. IEEE ICDM 2018.