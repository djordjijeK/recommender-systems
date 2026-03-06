import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))


import math
import torch
import logging
import pandas as pd

from typing import Dict, Tuple
from utils.data import get_data


logger = logging.getLogger("user-user-collaborative-filtering")


def leave_k_out_split(data_frame: pd.DataFrame, k: int = 5, random_state: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_shuffled = data_frame.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_shuffled['user_rating_count'] = df_shuffled.groupby('user_id')['user_id'].transform('count')
    df_shuffled['rating_index'] = df_shuffled.groupby('user_id').cumcount()

    test_condition = (df_shuffled['user_rating_count'] > k) & (df_shuffled['rating_index'] < k)

    df_train = df_shuffled.loc[~test_condition].drop(columns=['user_rating_count', 'rating_index'])
    df_test  = df_shuffled.loc[test_condition].drop(columns=['user_rating_count', 'rating_index'])
    
    return df_train, df_test


def get_user_similarities(centered_user_item_matrix: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(centered_user_item_matrix, p=2, dim=1, keepdim=True) + 1e-8
    normalized_matrix = centered_user_item_matrix / norm

    similarity_matrix = torch.mm(normalized_matrix, normalized_matrix.t()) 

    return similarity_matrix.fill_diagonal_(0)


def predict_rating(
    user_index: int, 
    movie_index: int, 
    centered_user_item_matrix: torch.Tensor, 
    user_item_matrix: torch.Tensor, 
    user_similarities: torch.Tensor,
    user_means: torch.Tensor        
) -> float:
    similarity_scores = user_similarities[user_index]
    movie_deviations = centered_user_item_matrix[:, movie_index]

    rated_mask = (user_item_matrix[:, movie_index] > 0).float()

    weighted_sum = torch.dot(similarity_scores * rated_mask, movie_deviations)
    sum_of_weights = torch.dot(torch.abs(similarity_scores), rated_mask)

    target_user_mean = user_means[user_index].item()

    if sum_of_weights < 1e-7:
        return target_user_mean

    prediction = target_user_mean + (weighted_sum / sum_of_weights)
    
    return prediction.item()


def evaluate_rating_predictions(
    data_frame_test: pd.DataFrame, 
    centered_matrix: torch.Tensor,
    user_item_matrix: torch.Tensor, 
    user_similarities: torch.Tensor, 
    user_means: torch.Tensor,
    user_id_to_index: Dict[str, int], 
    movie_id_to_index: Dict[str, int]
):
    squared_errors = []
    absolute_errors =[]

    for row in data_frame_test.itertuples():
        if row.user_id not in user_id_to_index or row.movie_id not in movie_id_to_index:
            continue

        user_index = user_id_to_index[row.user_id]
        movie_index = movie_id_to_index[row.movie_id]
        
        true_rating = float(row.rating)

        pred_rating = predict_rating(user_index, movie_index, centered_matrix, user_item_matrix, user_similarities, user_means)

        if not math.isnan(pred_rating):
            squared_errors.append((true_rating - pred_rating) ** 2)
            absolute_errors.append(abs(true_rating - pred_rating))

    rmse = math.sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else 0.0
    mae = sum(absolute_errors) / len(absolute_errors) if absolute_errors else 0.0

    return rmse, mae


if __name__ == "__main__":
    # 1. Load data
    data_frame = get_data()
    data_frame_train, data_frame_test = leave_k_out_split(data_frame)

    # 2. Map user and movie identifiers to indices
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(data_frame_train['user_id'].unique())}
    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(data_frame_train['movie_id'].unique())}

    data_frame_train['user_index'] = data_frame_train['user_id'].map(user_id_to_index)
    data_frame_train['movie_index'] = data_frame_train['movie_id'].map(movie_id_to_index)

    n_users = len(user_id_to_index)
    n_movies = len(movie_id_to_index)

    logger.info(f"Training set: {len(data_frame_train)} ratings, {n_users} users, {n_movies} movies")
    logger.info(f"Test set: {len(data_frame_test)} ratings")

    # 3. Create the user-movie matrix
    user_item_matrix = torch.zeros((n_users, n_movies))
    for row in data_frame_train.itertuples():
        user_id = int(row.user_index)
        movie_id = int(row.movie_index)
        user_item_matrix[user_id, movie_id] = float(row.rating)

    # 4. Compute user means and center the matrix by user means
    rated_mask = (user_item_matrix > 0).float()
    user_sums = user_item_matrix.sum(dim=1, keepdim=True)
    user_counts = rated_mask.sum(dim=1, keepdim=True)
    user_means = user_sums / user_counts 

    centered_user_item_matrix = (user_item_matrix - user_means) * rated_mask

    # 5. Calculate user similiarities
    user_similarities = get_user_similarities(centered_user_item_matrix)

    # 6. Evaluate model
    rmse, mae = evaluate_rating_predictions(
        data_frame_test,
        centered_user_item_matrix,
        user_item_matrix,
        user_similarities,
        user_means,
        user_id_to_index,
        movie_id_to_index
    )

    logger.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")