import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))


from load.movies import get_ratings


def build_dataset(min_interactions: int = 5) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[int]], int, int]:
    data = get_ratings()
 
    while True:
        n_before = len(data)
 
        movie_counts = data["movie_id"].value_counts()
        data = data[data["movie_id"].isin(
            movie_counts[movie_counts >= min_interactions].index
        )]
 
        user_counts = data["user_id"].value_counts()
        data = data[data["user_id"].isin(
            user_counts[user_counts >= min_interactions].index
        )]
 
        if len(data) == n_before:
            break
 
    # IDs start at 1 so that 0 is free for PAD.
    user_map  = {user_id: index + 1 for index, user_id in enumerate(sorted(data["user_id"].unique()))}
    movie_map = {movie_id: index + 1 for index, movie_id in enumerate(sorted(data["movie_id"].unique()))}

    data["user_id"] = data["user_id"].map(user_map)
    data["movie_id"] = data["movie_id"].map(movie_map)
 
    num_users = len(user_map)
    num_movies = len(movie_map)
 
    data = data.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
 
    # Rank 1 = most recent, rank 2 = second-most-recent, etc.
    data["_rank"] = (
        data.groupby("user_id")["timestamp"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
 
    test_sequences: dict[int, list[int]] = (
        data[data["_rank"] == 1]
        .groupby("user_id")["movie_id"]
        .apply(list)
        .to_dict()
    )
    valid_sequences: dict[int, list[int]]= (
        data[data["_rank"] == 2]
        .groupby("user_id")["movie_id"]
        .apply(list)
        .to_dict()
    )
 
    train_sequences: dict[int, list[int]] = (
        data[data["_rank"] >= 3]
        .sort_values("timestamp")
        .groupby("user_id")["movie_id"]
        .apply(list)
        .to_dict()
    )
 
    return train_sequences, valid_sequences, test_sequences, num_users, num_movies
