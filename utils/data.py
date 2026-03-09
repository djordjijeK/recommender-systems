import torch
import pandas as pd

from typing import Tuple
from pathlib import Path
from torch.utils.data import Dataset
from config import USER_ITEM_INTERACTIONS_DATA_PATH 


class UserItemInteractions(Dataset):

    def __init__(self, user_item_interaction_df: pd.DataFrame) -> None:
        super(UserItemInteractions, self).__init__()

        self._user_item_interactions = user_item_interaction_df

    
    def __len__(self):
        return self._user_item_interactions.shape[0]
    

    def __getitem__(self, index: int) -> Tuple[int, int, float]:
        user_item_interaction = self._user_item_interactions.iloc[index]

        return (
            int(user_item_interaction.user_id), 
            int(user_item_interaction.item_id), 
            float(user_item_interaction.rating)
        )


class UserItemMatrix(Dataset):

    def __init__(self, user_item_interaction_df: pd.DataFrame, num_users: int, num_items: int) -> None:
        super(UserItemMatrix, self).__init__()

        self._user_item_matrix = torch.zeros(num_users, num_items, dtype=torch.float32)
        for row in user_item_interaction_df.itertuples():
            user_id = int(row.user_id) - 1
            item_id = int(row.item_id) - 1

            self._user_item_matrix[user_id][item_id] = float(row.rating)

        self._user_item_interaction_mask = (self._user_item_matrix > 0.0).float().to(torch.float32)


    def __len__(self):
        return self._user_item_matrix.shape[0]   


    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        return index, self._user_item_matrix[index], self._user_item_interaction_mask[index]


def get_device(force_cpu_execution: bool = False) -> torch.device:
    if force_cpu_execution:
        return torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    return device


def get_user_item_interaction_data() -> pd.DataFrame:
    base_path = Path(__file__).resolve().parents[1]
    user_item_interaction_data_path = Path(f"{base_path}/{USER_ITEM_INTERACTIONS_DATA_PATH}")

    data_frame = pd.read_csv(
        filepath_or_buffer=user_item_interaction_data_path, 
        delimiter='\t', 
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    return data_frame


def leave_k_out_split(data_frame: pd.DataFrame, k: int = 10, random_state: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_shuffled = data_frame.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_shuffled['user_rating_count'] = df_shuffled.groupby('user_id')['user_id'].transform('count')
    df_shuffled['rating_index'] = df_shuffled.groupby('user_id').cumcount()

    test_condition = (df_shuffled['user_rating_count'] > k) & (df_shuffled['rating_index'] < k)

    df_train = df_shuffled.loc[~test_condition].drop(columns=['user_rating_count', 'rating_index'])
    df_test  = df_shuffled.loc[test_condition].drop(columns=['user_rating_count', 'rating_index'])
    
    return df_train, df_test
