import os
import pandas as pd


def get_dataloaders(dataset_root, batch_size=16, num_workers=4):
    train_parquet_path = os.path.join(dataset_root, "")

def get_dicts(csv_path):
    # Reading the csv file
    df = pd.read_parquet(csv_path)

    primary_labels = df['Label_Name'].values.tolist()
    primary_labels_unique = sorted(list(set(primary_labels)))  

    bird2label_dict = {}   # this dictionary will give integer 'label' when a 'bird' name is used in key
    label2bird_dict = {}   # this dictionary will give 'bird' name when integer 'label' is used in key

    for i, bird in enumerate(primary_labels_unique):
        bird2label_dict[bird] = i
        label2bird_dict[i] = bird
    
    return df, bird2label_dict, label2bird_dict