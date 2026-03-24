## SASRec: Self-Attentive Sequential Recommendation

An educational implementation of **SASRec** for sequential recommendation in PyTorch.

This project aims to capture the core idea and method faithfully rather than reproduce the original paper's codebase, preprocessing pipeline, or benchmark numbers exactly. The goal is understanding the model and implementing it in a way that is clear, readable, and true to the paper's core intuition.

### What is SASRec?

SASRec treats a user's interaction history as a sequence and learns to predict the next item using a causal (left-to-right) self-attention mechanism.

Unlike bidirectional models like BERT4Rec, SASRec is unidirectional: each position in the sequence can only attend to itself and earlier positions. This mirrors how recommendations work in practice - at prediction time, we only have past interactions to work with.

### The Core Intuition

A user's watch history can be viewed like a sentence:

- a movie is like a token
- a watch history is like a sentence
- recommendation becomes a next-token prediction problem

SASRec asks: *"Given the past interactions, what comes next?"*. It uses causal self-attention during training, which means each position can only see items that came before it. This allows the model to learn sequential patterns while preventing future information from leaking into predictions.

### Causal Masking

SASRec applies a causal (upper-triangular) mask to the attention scores, ensuring that position *i* can only attend to positions *0, 1, ..., i*. Combined with a padding mask that prevents attention to padded positions, this enforces a strict left-to-right information flow.

This is the key architectural difference from BERT4Rec, which uses no causal mask and allows each position to attend to the entire sequence.

### Training Objective

The model is trained with binary cross-entropy loss over positive and negative pairs. For every position in the training sequence:

- The positive target is the actual next item the user interacted with.
- A negative target is a randomly sampled item the user has not seen.

The loss pushes positive scores up and negative scores down. Padded positions are excluded from the loss.

### Inference

At recommendation time, the model processes the user's full interaction history and takes the output at the final position as the user's context representation:

`[item_1, item_2, ..., item_n] → context vector at position n`

A dot product between this context vector and all item embeddings produces a ranked list of candidate items.

### Evaluation

The model is evaluated with a standard ranking setup for next-item recommendation. For each user, the model ranks:

- 1 positive item - the last movie the user interacted with (held out)

- 100 negative items - randomly sampled movies the user did not interact with

Results:

- HR@1: 0.3841 - In 38.41% of cases, the model ranks the correct next movie at position 1 (best prediction).

- HR@5: 0.7177 - In 71.77% of cases, the correct movie appears within the top 5 predictions.

- NDCG@5: 0.5624 - The model not only places the correct item in the top 5, but often ranks it closer to the top positions.

- HR@10: 0.8114 - In 81.14% of cases, the correct movie appears within the top 10 predictions.

- NDCG@10: 0.5930 - Within the top 10, the correct item is typically ranked high rather than near the bottom.

- MRR: 0.5324 - On average, the correct item is ranked around 2nd place, indicating consistently strong ranking quality.

### Training

To train and evaluate the model, run the following from the project root:

```
python -m models.sasrec.train
```