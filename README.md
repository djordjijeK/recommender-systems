## Recommender Systems

Educational implementations of recommender system models in PyTorch, trained and evaluated on the [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) dataset.

The goal is to understand each model by implementing it from scratch in a way that is clear, readable, and true to the original paper's core intuition. 

### Models

- **User-User Collaborative Filtering** - Predicts ratings using adjusted cosine similarity between users. Aggregates ratings from similar users, weighted by similarity.

- **Item-Item Collaborative Filtering** - Same approach but computes similarity between movies instead of users. Generally more stable since item profiles change less than user profiles.

- **Matrix Factorization** - Learns latent user and movie embeddings via dot product. Trained with BCE loss and negative sampling.

- **Wide & Deep** *(2016)* - Linear "wide" component with hand-crafted cross-product features combined with a "deep" MLP over user/movie embeddings. ([Cheng et al., 2016](https://arxiv.org/abs/1606.07792))

- **Neural Collaborative Filtering (NCF)** *(2017)* — Combines a Generalized Matrix Factorization (element-wise product) branch with an MLP branch, fusing both through a linear output layer. ([He et al., 2017](https://arxiv.org/abs/1708.05031))

- **DeepFM** *(2017)* - Factorization Machine + Deep network sharing the same feature embeddings. The FM captures first and second-order interactions automatically, removing the need for manual feature engineering. ([Guo et al., 2017](https://arxiv.org/abs/1703.04247))

- **SASRec** *(2018)* - Unidirectional (causal) transformer for sequential recommendation. Predicts the next item using self-attention over the user's interaction history. ([Kang & McAuley, 2018](https://arxiv.org/abs/1808.09781))

- **BERT4Rec** *(2019)* - Bidirectional transformer using masked item prediction (Cloze task). Attends to both past and future context to reconstruct masked items. ([Sun et al., 2019](https://arxiv.org/abs/1904.06690))


### Results

All models are evaluated with the same ranking protocol: for each user, the model ranks 1 held-out positive item against 100 randomly sampled negatives.

| Model | HR@1 | HR@5 | NDCG@5 | HR@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|---|
| User-User CF | 0.0127 | 0.1018 | 0.0557 | 0.1998 | 0.0871 | 0.0779 |
| Item-Item CF | 0.0267 | 0.1439 | 0.0851 | 0.2555 | 0.1211 | 0.1029 |
| Matrix Factorization | 0.1811 | 0.4975 | 0.3435 | 0.6596 | 0.3961 | 0.3313 |
| Wide & Deep | 0.1614 | 0.4828 | 0.3270 | 0.6430 | 0.3790 | 0.3139 |
| NCF | 0.1538 | 0.4483 | 0.3049 | 0.6238 | 0.3615 | 0.2990 |
| DeepFM | 0.1692 | 0.4868 | 0.3315 | 0.6671 | 0.3898 | 0.3208 |
| SASRec | 0.3841 | 0.7177 | 0.5624 | 0.8114 | 0.5930 | 0.5324 |
| BERT4Rec | 0.3560 | 0.6760 | 0.5277 | 0.7808 | 0.5618 | 0.5017 |

## Dataset

[MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) contains ~1M ratings from ~6,040 users on ~3,706 movies. The data files (`users.dat`, `movies.dat`, `ratings.dat`) should be placed in the `data/` directory.

## Training

Each model can be trained and evaluated from the project root:

```bash
python -m models.user_user_cf
python -m models.item_item_cf
python -m models.matrix_factorization

python -m models.wide_and_deep.train
python -m models.ncf.train
python -m models.deep_fm.train
python -m models.sasrec.train
python -m models.bert4rec.train
```