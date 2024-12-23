import pandas as pd
from notebooks.bird_whisperer.datasets.dataset_utils import get_dicts

def prepare_data(train_data_path, test_data_path):
    train_df = pd.read_parquet(train_data_path)
    test_df = pd.read_parquet(test_data_path)
    bird2label_dict, label2bird_dict = get_dicts(pd.concat([train_df, test_df]))
    return bird2label_dict, label2bird_dict