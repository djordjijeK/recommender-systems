## BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer

An educational implementation of **BERT4Rec** for sequential recommendation in PyTorch.

This project aims to capture the core idea and method faithfully rather than reproduce the original paper's codebase, preprocessing pipeline, or benchmark numbers exactly. The goal is understanding the model and implementing it in a way that is clear, readable, and true to the paper's core intuition.

### What is BERT4Rec?

BERT4Rec treats a user's interaction history as a sequence and learns to predict missing items within that sequence using a bidirectional transformer.

In traditional sequential recommendation, models are often unidirectional: they only look at the items that came before the current one. BERT4Rec changes this by masking items in the sequence and training the model to recover them from the surrounding context.

That means each item can be understood using both:

- the items that came before it
- the items that came after it

This is the key idea that makes BERT4Rec interesting.

### The Core Intuition

A user's watch history can be viewed like a sentence:

- a movie is like a token
- a watch history is like a sentence
- recommendation becomes a masked sequence modeling problem

Instead of only asking: *“Given the past, what comes next?”*. BERT4Rec also asks: *“Given the surrounding sequence, what item belongs here?”*. BERT4Rec uses bidirectional self-attention during training, which means each position in a sequence can use information from both previous and following items. As a result, it can learn richer sequence patterns than models that only process the sequence from left to right.

### Masking Strategy

Training is based on a Cloze-style objective. A sequence such as:

`[Movie A, Movie B, Movie C, Movie D]`

might be transformed into:

`[Movie A, [MASK], Movie C, Movie D]`

and the model must predict the missing movie. This implementation uses two masking modes:

- Random masking: A percentage of items in the sequence are replaced with `[MASK]`. This teaches the model to infer missing items from surrounding context and builds general context understanding.

- Last-item masking: With some probability, the final item in the sequence is always masked. This is especially useful because it aligns training more closely with inference. At recommendation time, the model is asked to predict the next item at the end of the sequence, so explicitly training on last-item prediction helps reduce that mismatch.

### Training Objective

The model is trained to predict the original identity of masked items.

For positions that are not masked, the target label is `0`, which is ignored in the loss. For positions that are masked, the target label is the true item.

### Inference

At evaluation or recommendation time, the user's sequence is extended with a final `[MASK]` token:

`[item_1, item_2, ..., item_n, [MASK]]`

The model then predicts which item should occupy that final position. This makes next-item recommendation compatible with bidirectional training.

### Evaluation

The model is evaluated with a standard ranking setup for next-item recommendation. For each user, the model ranks:

- 1 positive item - the last movie the user interacted with (held out)

- 100 negative items - randomly sampled movies the user did not interact with

Results:

- HR@1: 0.3560 - In 36.29% of cases, the model ranks the correct next movie at position 1 (best prediction).

- HR@5: 0.6760 - In 67.53% of cases, the correct movie appears within the top 5 predictions.

- NDCG@5: 0.5277 - The model not only places the correct item in the top 5, but often ranks it closer to the top positions.

- HR@10: 0.7808 - In 77.15% of cases, the correct movie appears within the top 10 predictions.

- NDCG@10: 0.5618 - Within the top 10, the correct item is typically ranked high rather than near the bottom.

- MRR: 0.5017 - On average, the correct item is ranked around 2nd place, indicating consistently strong ranking quality.

### Training

To train the model, run the following from the project root:

```
python -m models.bert4rec.train
```