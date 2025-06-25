
import torch
from torch.utils.data import Dataset
from typing import Tuple



# Dataset from Task 1
class TaskDataset(Dataset):
    def __init__(self, transform=None):

        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index].convert('RGB')
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

# New Dataset to store the stolen embeddings with the self generated ones
class EmbeddingDataset(Dataset):
    def __init__(self, images= None, embeddings = None, transform=None):
        self.images = images.copy() if images else []
        self.embeddings = embeddings.copy() if embeddings else []
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].convert('RGB')
        if self.transform:
            img = self.transform(img)
        emb = torch.tensor(self.embeddings[idx])
        return img, emb
    
    def extend(self, new_images, new_embeddings):
        self.images.extend(new_images)
        self.embeddings.extend(new_embeddings)
      