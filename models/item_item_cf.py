import random
import torch
import logging
import pandas as pd

from config import *
from load.movies import get_ratings


logger = logging.getLogger("model:item-item-collaborative-filtering")


def leave_one_out_split(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = ratings.sort_values("timestamp")

    test = ratings.groupby("user_id").tail(1)
    train = ratings.drop(test.index)

    return train, test


def build_user_movie_matrix(train: pd.DataFrame, num_users: int, num_movies: int) -> torch.Tensor:
    matrix = torch.zeros(num_users, num_movies)

    user_ids = torch.tensor(train["user_id"].values, dtype=torch.long)
    movie_ids = torch.tensor(train["movie_id"].values, dtype=torch.long)
    ratings = torch.tensor(train["rating"].values, dtype=torch.float32)

    matrix[user_ids, movie_ids] = ratings

    return matrix


def compute_user_means(user_movie_matrix: torch.Tensor) -> torch.Tensor:
    rated_mask = (user_movie_matrix > 0).float()
    user_sums = user_movie_matrix.sum(dim=1)
    user_counts = rated_mask.sum(dim=1).clamp(min=1)
    
    return user_sums / user_counts


def center_matrix(user_movie_matrix: torch.Tensor, user_means: torch.Tensor) -> torch.Tensor:
    rated_mask = (user_movie_matrix > 0).float()
    return (user_movie_matrix - user_means.unsqueeze(1)) * rated_mask


def get_movie_similarities(centered_user_movie_matrix: torch.Tensor) -> torch.Tensor:
    movie_vectors = centered_user_movie_matrix.t()

    norms = movie_vectors.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
    normalized = movie_vectors / norms
    similarities = normalized @ normalized.t()
    similarities.fill_diagonal_(0)

    return similarities


def score_movie(
    user_id: int,
    movie_id: int,
    centered_user_movie_matrix: torch.Tensor,
    user_movie_matrix: torch.Tensor,
    movie_similarities: torch.Tensor,
    user_means: torch.Tensor,
) -> float:
    similarities = movie_similarities[movie_id]
    deviations = centered_user_movie_matrix[user_id]
    rated_mask = (user_movie_matrix[user_id] > 0).float()

    weighted_sum = torch.dot(similarities * rated_mask, deviations)
    weight_total = torch.dot(similarities.abs(), rated_mask)

    if weight_total < 1e-7:
        return user_means[user_id].item()

    return (user_means[user_id] + weighted_sum / weight_total).item()


def evaluate(
    test_data: pd.DataFrame,
    train_data: pd.DataFrame,
    centered_user_movie_matrix: torch.Tensor,
    user_movie_matrix: torch.Tensor,
    movie_similarities: torch.Tensor,
    user_means: torch.Tensor,
    all_movie_ids: list[int],
    num_negatives: int = 100,
) -> dict[str, float]:
    rng = random.Random(1347)
    train_user_movies = train_data.groupby("user_id")["movie_id"].apply(set).to_dict()

    all_ranks = []
    for row in test_data.itertuples():
        user_id, positive_movie_id = row.user_id, row.movie_id
        seen = train_user_movies.get(user_id, set()) | {positive_movie_id}

        negative_movies: list[int] = []
        while len(negative_movies) < num_negatives:
            candidate = rng.choice(all_movie_ids)
            if candidate not in seen:
                negative_movies.append(candidate)
                seen.add(candidate)

        candidates = [positive_movie_id] + negative_movies
        scores = [
            score_movie(int(user_id), int(movie_id), centered_user_movie_matrix, user_movie_matrix, movie_similarities, user_means)
            for movie_id in candidates
        ]

        positive_score = scores[0]
        rank = sum(1 for score in scores[1:] if score >= positive_score) + 1
        all_ranks.append(rank)

    ranks = torch.tensor(all_ranks, dtype=torch.float)

    metrics: dict[str, float] = {}
    for k in (1, 5, 10):
        hit = (ranks <= k).float()
        metrics[f"HR@{k}"] = hit.mean().item()
        if k != 1:
            metrics[f"NDCG@{k}"] = (hit / torch.log2(ranks + 1)).mean().item()
    metrics["MRR"] = (1.0 / ranks).mean().item()

    return metrics


if __name__ == "__main__":
    ratings = get_ratings()

    num_users = ratings["user_id"].max() + 1
    num_movies = ratings["movie_id"].max() + 1

    train_data, test_data = leave_one_out_split(ratings)

    # 1. Build user-movie matrix and center by user means
    user_movie_matrix = build_user_movie_matrix(train_data, num_users, num_movies)
    user_means = compute_user_means(user_movie_matrix)
    centered_matrix = center_matrix(user_movie_matrix, user_means)

    # 2. Compute item-item similarities
    movie_similarities = get_movie_similarities(centered_matrix)

    # 3. Evaluate
    all_movie_ids = ratings["movie_id"].unique().tolist()

    metrics = evaluate(test_data, train_data, centered_matrix, user_movie_matrix, movie_similarities, user_means, all_movie_ids)
    
    metrics_string = " | ".join(f"{metric}: {value:.4f}" for metric, value in metrics.items())
    logger.info(f"Test metrics - {metrics_string}")
