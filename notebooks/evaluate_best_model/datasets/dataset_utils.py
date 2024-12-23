import os
import pandas as pd
from torch.utils.data import DataLoader
from .dataset import XenoCantoDataset

def get_train_test_data(test_df, bird2label_dict, seed=42):
    lambda_function = lambda bird: bird2label_dict[bird]

    test_df.insert(1, "labels", None) # adding new column "labels" at index=1. All values will be None in this column
    test_df["labels"] = test_df["species"].apply(lambda_function) # filling the "labels" column with integer labels

    # shuffling the dataframes
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    audio_files_paths_test = test_df["spectogram_file"].values.tolist()
    labels_test = test_df["labels"].values.tolist()

    labels_unique = sorted(list(set(labels_test)))

    return audio_files_paths_test, labels_test, labels_unique

def get_dataloaders(dataset_root, batch_size, num_workers, test_parquet_name, with_augmented=True):
    test_parquet_path = os.path.join(dataset_root, test_parquet_name)

    test_df = pd.read_parquet(test_parquet_path)

    if not with_augmented:
        test_df = test_df[test_df["augmented"] == False]

    bird2label_dict, label2bird_dict = get_dicts(pd.concat([test_df]))

    assert len(bird2label_dict) == len(label2bird_dict), "Bird2Label and Label2Bird dictionaries are not equal"

    audio_files_paths_test, labels_test, labels_unique = get_train_test_data(test_df, bird2label_dict)

    test_spectograms_folder = os.path.join(dataset_root, "spectograms", "test")
    test_dataset = XenoCantoDataset(audio_files_paths_test, labels_test, test_spectograms_folder, bird2label_dict, label2bird_dict)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader, labels_unique

def get_dicts(df):
    primary_labels = df["species"].values.tolist()
    primary_labels_unique = sorted(list(set(primary_labels)))

    bird2label_dict = {} # this dictionary will give integer "label" when a "bird" name is used in key
    label2bird_dict = {} # this dictionary will give "bird" name when integer "label" is used in key

    for i, bird in enumerate(primary_labels_unique):
        bird2label_dict[bird] = i
        label2bird_dict[i] = bird
    
    return bird2label_dict, label2bird_dict