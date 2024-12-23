import os
import pandas as pd
from torch.utils.data import DataLoader
from .dataset import XenoCantoDataset

def get_dataloaders(dataset_root, batch_size, num_workers, train_parquet_name, test_parquet_name, with_augmented=True):
    train_parquet_path = os.path.join(dataset_root, train_parquet_name)
    test_parquet_path = os.path.join(dataset_root, test_parquet_name)

    train_df = pd.read_parquet(train_parquet_path)
    test_df = pd.read_parquet(test_parquet_path)

    if not with_augmented:
        train_df = train_df[train_df["augmented"] == False]
        test_df = test_df[test_df["augmented"] == False]

    bird2label_dict, label2bird_dict = get_dicts(pd.concat([train_df, test_df]))

    audio_files_paths_test = test_df["spectogram_file"].values.tolist()
    labels_test = test_df["labels"].map(bird2label_dict).values.tolist()

    test_spectograms_folder = os.path.join(dataset_root, "spectograms", "test")
    test_dataset = XenoCantoDataset(audio_files_paths_test, labels_test, test_spectograms_folder, bird2label_dict, label2bird_dict)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return None, test_loader, label2bird_dict

def get_dicts(df):
    primary_labels = df["species"].values.tolist()
    primary_labels_unique = sorted(list(set(primary_labels)))

    bird2label_dict = {bird: i for i, bird in enumerate(primary_labels_unique)}
    label2bird_dict = {i: bird for bird, i in bird2label_dict.items()}

    return bird2label_dict, label2bird_dict
