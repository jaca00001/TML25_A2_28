import pickle
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple
import random
import requests
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import json
import io
import sys
import base64
import os
import timm
from torchvision.models import resnet50
import torch.optim as optim 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SEED = 27970870
PORT = 9817
TOKEN = "08392413" 

class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

class ResNet50Encoder1024(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-1]) 
        self.linear = nn.Linear(2048, 1024)
        #for param in self.backbone.parameters():
        #    param.requires_grad = False


    def forward(self, x):
        x = self.backbone(x)              
        x = x.view(x.size(0), -1)         
        x = self.linear(x)                
        return x

class EmbeddingDataset(Dataset):

    def __init__(self, embs, images, transform):
        self.embs = embs
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        emb1, emb2, emb3, emb4, emb5 = self.embs[idx]
        if self.transform:
            image = self.transform(image)
        emb_list = (
            torch.tensor(emb1, dtype=torch.float32),
            torch.tensor(emb2, dtype=torch.float32),
            torch.tensor(emb3, dtype=torch.float32),
            torch.tensor(emb4, dtype=torch.float32),
            torch.tensor(emb5, dtype=torch.float32),
        )
        return image, emb_list

class TestDataset(Dataset):

    def __init__(self, images, representations, transform):
        self.images = images
        self.transform = transform
        self.representations = representations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        reps = torch.tensor(self.representations[idx], dtype=torch.float32)
        return image, reps


# ----------- Training loop -----------
def train():
    best_val_loss = float('inf')
    for epoch in range(50):  # 10 epochs
        model.train()
        train_loss = 0.0
        print(f"\nEpoch {epoch+1} Training:")

        for i, (image, target_features) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            image = image.to(device)

            feature1, feature2, feature3, feature4, feature5 = target_features
            feature1 = feature1.to(device)
            feature2 = feature2.to(device)
            feature3 = feature3.to(device)
            feature4 = feature4.to(device)
            feature5 = feature5.to(device)
            features = [feature1, feature2, feature3, feature4, feature5]
            #features = features.to(device)

            optimizer.zero_grad()
            predicted_feature = model(image)

            loss_mse = 0
            loss_cosine = 0 
            for j in range(len(features)):
                loss_mse = loss_mse + mse_criterion(predicted_feature, features[j])
                loss_cosine = loss_cosine + (1-cos_criterion(predicted_feature, features[j])).mean()
            loss = loss_mse + loss_cosine
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"  [Batch {i+1}/{len(train_loader)}] MSE + Cosine Loss: {loss.item():.4f}")

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image, target_features in validation_loader:  
                image = image.to(device)
                feature1, feature2, feature3, feature4, feature5 = target_features
                feature1 = feature1.to(device)
                feature2 = feature2.to(device)
                feature3 = feature3.to(device)
                feature4 = feature4.to(device)
                feature5 = feature5.to(device)
                features = [feature1, feature2, feature3, feature4, feature5]
        
                predicted_feature = model(image)
        
                loss_mse = 0
                loss_cosine = 0 
                for j in range(len(features)):
                    loss_mse += mse_criterion(predicted_feature, features[j])
                    loss_cosine += (1 - cos_criterion(predicted_feature, features[j])).mean()
                loss = loss_mse + loss_cosine
                val_loss += loss.item()
        val_loss /= len(validation_loader)
        scheduler.step()

        print(f"Epoch {epoch+1} Summary: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_encoder_model.pt")
            print(f"Saved model at epoch {epoch+1} with val loss {val_loss:.4f}")


def get_dataloader():
    with open('original_out.pickle', 'rb') as handle:
        out1 = pickle.load(handle)

    with open('augmented_out.pickle', 'rb') as handle:
        out2 = pickle.load(handle)

    emb1, emb2 = out1["original_v1"],out1["original_v2"]
    emb3, emb4, emb5 = out2["flip"], out2["rotate"], out2["color"] 

    # Split train/val
    indices = list(range(len(dataset.imgs)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_images = [dataset.imgs[i] for i in train_idx]
    validation_images = [dataset.imgs[i] for i in val_idx]
    train_features = [(emb1[idx],emb2[idx],emb3[idx],emb4[idx],emb5[idx]) for idx in train_idx]
    validation_features = [(emb1[idx],emb2[idx],emb3[idx],emb4[idx],emb5[idx]) for idx in val_idx]

    # Datasets and loaders
    train_dataset = EmbeddingDataset(train_features, train_images, transform=transform)
    validation_dataset = EmbeddingDataset(validation_features, validation_images, transform=transform)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128)
    validation_loader = DataLoader(validation_dataset, shuffle=False, batch_size=128)

    return train_loader, validation_loader

def model_stealing(images, port):
    endpoint = "/query"
    url = f"http://34.122.51.94:{port}" + endpoint
    image_data = []
    for img in images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        image_data.append(img_base64)

    payload = json.dumps(image_data)
    response = requests.get(url, files={"file": payload}, headers={"token": TOKEN})

    if response.status_code == 200:
        return response.json()["representations"]
    else:
        raise Exception(
            f"Model stealing failed. Code: {response.status_code}, content: {response.json()}"
        )

def test():
    batch_imgs = dataset.imgs[2500:3500]
    total_mse = 0.0
    '''
    representations = model_stealing(batch_imgs, port=PORT)
    with open('out1.pickle', 'wb') as handle:
        pickle.dump(representations, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''
    with open('out1.pickle', 'rb') as handle:
         representations = pickle.load(handle)
    
    test_dataset = TestDataset(batch_imgs,representations, transform=transform)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=4)
    for image, target_feature in test_dataloader:
        image = image.to(device)
        predicted_feature = model(image)
        target_feature = target_feature.to(device)
        loss = mse_criterion(predicted_feature, target_feature)
        total_mse = total_mse + loss.item()
    print(f"Epoch MSE on test data : {total_mse/250}")



    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    submission = True
    training = False
    testing = True
    # Load dataset with images
    dataset: TaskDataset = torch.load("ModelStealingPub.pt", weights_only=False)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2980, 0.2962, 0.2987],
                            std=[0.2886, 0.2875, 0.2889])
    ])
    
    if training:
        # Model, optimizer, contrastive loss
        model = ResNet50Encoder1024().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
        mse_criterion = nn.MSELoss()
        cos_criterion = nn.CosineSimilarity(dim=1) 
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        train_loader, validation_loader = get_dataloader()
        train()
    
    if testing:
        mse_criterion = nn.MSELoss()
        model = ResNet50Encoder1024().to(device)
        model.load_state_dict(torch.load("best_encoder_model.pt", map_location=device))
        model.eval()
        test()
    
    
    if submission:
        path = "team28_submission.onnx"
        torch.onnx.export(
            model.cpu(),
            torch.randn(1, 3, 32, 32),
            path,
            export_params=True,
            input_names=["x"],
        )
        
        with open(path, "rb") as f:
            model = f.read()
            try:
                stolen_model = ort.InferenceSession(model)
            except Exception as e:
                raise Exception(f"Invalid model, {e=}")
            try:
                out = stolen_model.run(
                    None, {"x": np.random.randn(1, 3, 32, 32).astype(np.float32)}
                )[0][0]
            except Exception as e:
                raise Exception(f"Some issue with the input, {e=}")
            assert out.shape == (1024,), "Invalid output shape"
        response = requests.post("http://34.122.51.94:9090/stealing", files={"file": open(path, "rb")}, headers={"token": TOKEN, "seed": str(SEED)})
        print(response.json())
    


