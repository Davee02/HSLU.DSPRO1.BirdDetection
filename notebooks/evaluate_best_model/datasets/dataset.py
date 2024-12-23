import os
import torch
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
    
    def __getitem__(self, index):
        audio_path = self.audio_files_path[index]
        audio_full_path = os.path.join(self.main_folder_path, audio_path)

        mel = torch.load(audio_full_path)
        label = torch.tensor(self.labels[index])
        
        return mel, label, self.label2bird_dict[label.item()]