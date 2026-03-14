import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))


import math
import torch
import logging
import pandas as pd

from typing import Dict
from utils.datav1 import get_user_item_interaction_data, leave_k_out_split, UserItemMatrix


logger = logging.getLogger("model:item-item-collaborative-filtering")


def get_item_similarities(centered_user_item_matrix: torch.Tensor) -> torch.Tensor:
    item_matrix = centered_user_item_matrix.t() 

    norm = torch.norm(item_matrix, p=2, dim=1, keepdim=True) + 1e-8
    normalized_matrix = item_matrix / norm

    similarity_matrix = torch.mm(normalized_matrix, normalized_matrix.t()) 

    return similarity_matrix.fill_diagonal_(0)


def predict_rating(
    user_index: int,
    item_index: int,
    centered_user_item_matrix: torch.Tensor,
    user_item_matrix: torch.Tensor,
    item_similarities: torch.Tensor,
    user_means: torch.Tensor           
) -> float:
    similarity_scores = item_similarities[item_index - 1]       
    user_deviations = centered_user_item_matrix[user_index - 1, :]         

    rated_mask = (user_item_matrix[user_index - 1, :] > 0).float()  

    weighted_sum = torch.dot(similarity_scores * rated_mask, user_deviations)
    sum_of_weights = torch.dot(torch.abs(similarity_scores), rated_mask)

    target_user_mean = user_means[user_index - 1].item()

    if sum_of_weights < 1e-7:
        return target_user_mean

    prediction = target_user_mean + (weighted_sum / sum_of_weights)

    return prediction.item()


def evaluate_rating_predictions(
    data_frame_test: pd.DataFrame,
    centered_user_item_matrix: torch.Tensor,
    user_item_matrix: torch.Tensor,
    item_similarities: torch.Tensor,
    user_means: torch.Tensor
):
    squared_errors = []
    absolute_errors = []

    for row in data_frame_test.itertuples():
        user_index = int(row.user_id)
        item_index = int(row.item_id)

        true_rating = float(row.rating)

        predicted_rating = predict_rating(
            user_index,
            item_index,
            centered_user_item_matrix,
            user_item_matrix,
            item_similarities,
            user_means
        )

        if not math.isnan(predicted_rating):
            squared_errors.append((true_rating - predicted_rating) ** 2)
            absolute_errors.append(abs(true_rating - predicted_rating))

    rmse = math.sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else 0.0
    mae = sum(absolute_errors) / len(absolute_errors) if absolute_errors else 0.0

    return rmse, mae


if __name__ == "__main__":
    # 1. Load data
    data_frame = get_user_item_interaction_data()

    num_users = len(data_frame.user_id.unique())
    num_items = len(data_frame.item_id.unique())

    data_frame_train, data_frame_test = leave_k_out_split(data_frame, random_state=1347)

    user_item_interactions_dataset = UserItemMatrix(data_frame_train, num_users, num_items)

    # 2. Compute user means and center the matrix by user means
    user_item_matrix = user_item_interactions_dataset._user_item_matrix

    rated_mask = (user_item_matrix > 0).float()
    user_sums = user_item_matrix.sum(dim=1, keepdim=True)
    user_counts = rated_mask.sum(dim=1, keepdim=True)
    user_means = user_sums / user_counts

    centered_user_item_matrix = (user_item_matrix - user_means) * rated_mask

    # 3. Train & evaluate model
    item_similarities = get_item_similarities(centered_user_item_matrix)

    rmse, mae = evaluate_rating_predictions(
        data_frame_test,
        centered_user_item_matrix,
        user_item_matrix,
        item_similarities,
        user_means
    )

    logger.info(f"Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}")
