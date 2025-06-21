import requests
import torch
import onnxruntime as ort
import numpy as np
import json
import io
import sys
import base64
from torch.utils.data import Dataset,DataLoader
from typing import Tuple
import pickle
import os
import torchvision.models as models
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import time
import kornia.augmentation as K
import torch.nn as nn

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2980, 0.2962, 0.2987],
                         std=[0.2886, 0.2875, 0.2889]),

])

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

#create this to better store the stolen embeddings
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
      
  
class SmallResNet18(nn.Module):
    def __init__(self, output_dim=1024, dropout_p=0.3, pretrained=False):

        super().__init__()

 
        self.backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        #reduce to work on smaller image sizes as well, might reduce dimesions too quicly otherwise
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.backbone.maxpool = nn.Identity()


        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])

        self.head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        
        self.out = nn.Linear(512, output_dim)
        
        

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = x + self.head(x)  
        return self.out(x)




import kornia.augmentation as K

AUGMENT = nn.Sequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomRotation(degrees=20.0),                     
    K.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2, p=0.7),
    K.RandomAffine(degrees=15, translate=0.2, scale=(0.8, 1.2), shear=10.0, p=0.7), 
    K.RandomGrayscale(p=0.2),
    K.RandomErasing(p=0.5, scale=(0.02, 0.3), ratio=(0.3, 3.3)),                               
).cuda()

from torchvision import transforms

AUGMENT_SINGLE = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.3), ratio=(0.3, 3.3)),
])


def augment_image(images):
    return AUGMENT(images)

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

def augment_image_single(image):
    image = to_tensor(image)   
    image = AUGMENT_SINGLE(image) 
    image = to_pil(image)     
    return image

def bayesian_predictions(model, X, estimators=10, device='cuda'):
    model.train()

    with torch.no_grad():
     
        views = torch.cat([augment_image(X) for _ in range(estimators)], dim=0).to(device)  
        out = model(views)  

    B = X.size(0)
    out = out.view(estimators, B, -1)

    std = out.std(dim=0).mean(dim=1)  
    return std

from torch.utils.data import Subset

def predict_next(model, dataset, samples=10, k=4000, subset_ratio=0.2, used_indices=None):
    # Determine unused indices
    all_indices = set(range(len(dataset)))
    used_indices = set(used_indices) if used_indices is not None else set()
    unused_indices = list(all_indices - used_indices)

    # Sample a random subset from unused indices
    subset_size = int(subset_ratio * len(dataset))
    subset_indices = np.random.choice(unused_indices, size=min(subset_size, len(unused_indices)), replace=False)

    # Create subset and loader
    subset = Subset(dataset, subset_indices)
    subset_loader = DataLoader(subset, batch_size=256, shuffle=False)

    # Collect uncertainty scores (std of Bayesian predictions)
    stds = []
    for _, imgs, _ in tqdm(subset_loader):
        imgs = imgs.cuda()
        batch_stds = bayesian_predictions(model, imgs, estimators=samples)
        stds.append(batch_stds.cpu())

    stds = torch.cat(stds)
    
    # Map stds back to dataset indices
    stds_np = stds.numpy()
    subset_indices = np.array(subset_indices)
    topk_local = np.argsort(stds_np)[-k:]
    topk_indices = subset_indices[topk_local]

    # Class-balanced sampling
    labels = np.array(dataset.labels)
    label_counts = np.bincount(labels)
    label_probs = label_counts / label_counts.sum()
    topk_labels = labels[topk_indices]
    weights = label_probs[topk_labels]
    weights /= weights.sum()

    sampled_indices = np.random.choice(topk_indices, size=1000, replace=False, p=weights)
    return sampled_indices

def snn_contrastive_loss(embeddings_a, embeddings_b, T=0.5):

    a = F.normalize(embeddings_a, dim=1) 
    b = F.normalize(embeddings_b, dim=1)  

    sim_matrix = torch.matmul(a, b.t())  

    scaled_sim = sim_matrix / T
   
    numerator = torch.exp(torch.diag(scaled_sim))  
   
    denominator = torch.sum(torch.exp(scaled_sim), dim=1) 

    loss_per_sample = -torch.log(numerator / denominator)

    return loss_per_sample.mean()
   


import matplotlib.pyplot as plt

def train(model, loader, epochs, round):
    
    # Freeze or unfreeze feature extractor depending on the round
    if round > 3:
        for param in model.feature_extractor.parameters():
            param.requires_grad = True
    else:
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

    # Optimizer only for trainable parameters
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )

    model.cuda()
    criterion = nn.MSELoss()
    epoch_losses = []
    epoch_l2_dists = []

    pbar = tqdm(range(epochs), leave=False)
    for epoch in pbar:
        model.train()
        total_loss = 0
        total_l2 = 0
        count = 0

        alpha = (epoch / epochs) * 0.5

        for imgs, embeddings in loader:
            imgs, embeddings = imgs.cuda(), embeddings.cuda()
            view_1 = augment_image(imgs).cuda()

            pred_embeddings_1 = model(view_1)

            mse_loss = criterion(pred_embeddings_1, embeddings)
            cos_loss = F.cosine_embedding_loss(pred_embeddings_1, embeddings, torch.ones(pred_embeddings_1.size(0)).cuda())

            weighted_mse = (1 - alpha) * mse_loss
            weighted_cos = alpha * cos_loss
            loss = weighted_mse + weighted_cos

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_l2 += F.mse_loss(pred_embeddings_1, embeddings, reduction='sum').sqrt().item()
            count += imgs.size(0)

        avg_loss = total_loss / count
        avg_l2 = total_l2 / count
        epoch_losses.append(avg_loss)
        epoch_l2_dists.append(avg_l2)
        pbar.set_postfix({'L2 dist': f'{avg_l2:.4f}'})

    torch.save(model.state_dict(), f"model_out/surrogate_model{round}.pth")

    # Plot training loss
    plt.figure(figsize=(6, 4))
    plt.plot(epoch_losses, label="Avg Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss - Round {round}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"model_out/loss_plot_round{round}.png")
    plt.close()

    return model

def attack(dataset, rounds, device='cuda'):
   
    selected_indices = np.array([])
    training_data = EmbeddingDataset(transform=transform)
    used_indices = []
    
    surrogate_model = SmallResNet18(output_dim=1024, pretrained=True)
    
    for i in range(rounds):
          
        if selected_indices.size == 0:
            selected_indices = np.random.permutation(1000)
        else:
            print(embeddings_sim())
            if i % 5 == 0:
                loader = DataLoader(training_data, batch_size=128, shuffle=True)
            else:   
                loader = DataLoader(current_data, batch_size=128, shuffle=True)

            if len(dataset) - len(used_indices) < 1000:
               loader = DataLoader(training_data, batch_size=128, shuffle=True)
               surrogate_model  = train(surrogate_model, loader,100, i)
               
               break
            
            surrogate_model  = train(surrogate_model, loader,50, i)
            selected_indices = predict_next(surrogate_model, dataset, used_indices=used_indices,samples=10)
            used_indices.extend(selected_indices.tolist())
            
      
        images = [augment_image_single(dataset.imgs[idx]) for idx in selected_indices]  
        
        
        embeddings = model_stealing(images, port=PORT)  
        
        with open(f'api_out/embeddings{i}.pickle', 'wb') as handle:
            pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
        current_data = EmbeddingDataset(transform=transform)
        current_data.extend(images,embeddings)
        training_data.extend(images,embeddings)


import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

def embeddings_sim(dataset):
    # Load CIFAR-10 test set
    transform = transforms.Compose([
        transforms.ToTensor(),  # converts to [0,1]
    ])
    
    
    time.sleep(60)
    selected = np.random.permutation(len(dataset))[1000:2000]
    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()
    
 
    images = [augment_image_single(to_pil(dataset[i][1])) for i in selected]  
    embeddings = torch.tensor(model_stealing(images, port=PORT))
   
   
    selected = np.random.permutation(len(dataset))[:1000]
    images = [augment_image_single(to_pil(dataset[i][1])) for i in selected] 
    time.sleep(60)
    
    embeddings2 = torch.tensor(model_stealing(images, port=PORT))

    time.sleep(60)
    
    # Normalize
    norm1 = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
    norm2 = embeddings2 / (embeddings2.norm(dim=1, keepdim=True) + 1e-8)

    # Cosine similarity
    similarities = (norm1 * norm2).sum(dim=1).cpu().numpy()
    return similarities.mean()

def upload(model_path):
    path = 'dummy_submission.onnx'

    model = SmallResNet18().cuda()

    model.load_state_dict(torch.load(model_path,weights_only=False))
    model.eval() 
    print("Model loaded!")
        
    
    torch.onnx.export(
        model.cpu(),
        torch.randn(1, 3, 32, 32),
        path,
        export_params=True,
        input_names=["x"],
    )

    #### Tests ####

    # (these are being ran on the eval endpoint for every submission)
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

    # Send the model to the server
    response = requests.post("http://34.122.51.94:9090/stealing", files={"file": open(path, "rb")}, headers={"token": TOKEN, "seed": SEED})
    print(response.json())

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
        representation = response.json()["representations"]
        return representation
    else:
        raise Exception(
            f"Model stealing failed. Code: {response.status_code}, content: {response.json()}"
        )

def new_api():
    response = requests.get("http://34.122.51.94:9090" + "/stealing_launch", headers={"token": TOKEN})
    answer = response.json()

    print(answer)  # {"seed": "SEED", "port": PORT}

    # Save to file
    with open("data.txt", "w") as f:
        json.dump(answer, f)

    
    if 'detail' in answer:
        sys.exit(1)
        


# load the dataset
dataset = torch.load("ModelStealingPub.pt",weights_only=False)

# change the labels to be 0,1,2,...,n
original_labels = np.array(dataset.labels)
unique_labels = np.unique(original_labels)
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=0)}
new_labels = np.array([label_mapping[label] for label in original_labels])
dataset.labels = new_labels.tolist()

# load the transform
dataset.transform = transform

TOKEN = "08392413"
SEED = "41777894"
PORT = "9817"



print(embeddings_sim(dataset))

# new_api()#  # 11:28 -> 15:28
attack(dataset, rounds=400)  
# upload("model_out/surrogate_model14.pth") # dont know anymore, maybe ensemble or other model,  better general , more data, look at space occupied -> 10:57, 

