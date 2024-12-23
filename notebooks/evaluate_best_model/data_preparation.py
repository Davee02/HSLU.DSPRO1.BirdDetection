import pandas as pd
from notebooks.bird_whisperer.datasets.dataset_utils import get_dicts

def prepare_data(train_data_path, test_data_path):
    train_df = pd.read_parquet(train_data_path)
    test_df = pd.read_parquet(test_data_path)
    bird2label_dict, label2bird_dict = get_dicts(pd.concat([train_df, test_df]))
    return bird2label_dict, label2bird_dict

def load_evaluation_data(evaluation_data_path):
    eval_df = pd.read_parquet(evaluation_data_path)
    bird2label_dict, label2bird_dict = get_dicts(eval_df)
    true_labels = eval_df['label'].map(bird2label_dict).tolist()
    evaluation_data = eval_df['features'].tolist()  # Assuming features are precomputed
    return evaluation_data, true_labels, bird2label_dict, label2bird_dict
