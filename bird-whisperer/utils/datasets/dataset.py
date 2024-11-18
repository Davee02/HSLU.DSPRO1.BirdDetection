from torch.utils.data import Dataset

class XenoCantoDataset(Dataset):

    def __init__(self, audio_files_path, labels, main_folder_path, bird2label_dict, label2bird_dict):
        self.audio_files_path = audio_files_path
        self.labels = labels
        self.main_folder_path = main_folder_path
        self.bird2label_dict = bird2label_dict
        self.label2bird_dict = label2bird_dict

    def __len__(self):
        return len(self.audio_files_path)