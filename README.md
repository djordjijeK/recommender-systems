### [Wide & Deep Learning for Recommender Systems (Cheng et al., 2016)](ranking/wide_and_deep.py)

An implementation of the Wide & Deep architecture (Cheng et al., 2016) applied to the MovieLens 1M dataset for movie rating prediction.

Recommender systems need both memorisation (exploiting known, specific patterns) and generalisation (transferring knowledge to unseen combinations). 
Wide-only models excel at the former, deep-only models at the latter. 
Wide & Deep jointly trains both under a single loss so each covers the other's weakness.

- Wide component - a linear layer over one-hot features and cross-product interaction terms (gender × genre, occupation × genre). Handles memorisation.


- Deep component - categorical features (user, movie, gender, age, occupation) passed through embedding tables, concatenated with continuous features, and fed through a ReLU MLP. Handles generalisation.


- Output - the scalar outputs of both components are summed and trained jointly end-to-end.


```
Train MSE: 0.7365 (RMSE: 0.8582) | Validation MSE: 0.8796 (RMSE: 0.9379)
Test MSE: 0.9077 | Test RMSE: 0.9527
```


### [Neural Collaborative Filtering (He et al., 2017)](ranking/neural_collaborative_filtering.py)

An implementation of the Neural Collaborative Filtering architecture (He et al., 2017) applied to the MovieLens 1M dataset for movie rating prediction.

Traditional matrix factorisation models estimate user–item affinity through a simple latent dot product, which is efficient but limited in expressive power.
Neural Collaborative Filtering replaces this fixed interaction function with a learnable neural architecture, allowing the model to capture both linear and non-linear patterns in user–item behaviour.


- GMF component - learns user and item embeddings and combines them with element-wise multiplication. Captures linear interaction signals in the spirit of matrix factorisation.


- MLP component - learns separate user and item embeddings, concatenates them, and feeds them through a stack of ReLU layers with dropout. Captures higher-order, non-linear interaction patterns.


- Output - the representations from the GMF and MLP branches are concatenated and passed through a final linear layer, with additional learned user and item bias terms added to produce the rating prediction.

```
Train MSE: 0.7804 (RMSE: 0.8834) | Validation MSE: 0.9566 (RMSE: 0.9780)
Test MSE: 0.9921 | Test RMSE: 0.9961
```