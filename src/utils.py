
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import kornia.augmentation as K


# Augmentations specified for the images
AUGMENT = nn.Sequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomRotation(degrees=40),                                                  
).cuda()

AUGMENT_SINGLE = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(40),
])

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()


# One for Batch one for single images
def augment_image(images):
    return AUGMENT(images)


def augment_image_single(image):
    image = to_tensor(image)   
    image = AUGMENT_SINGLE(image) 
    image = to_pil(image)     
    return image

# Use MC-Droput to estimate the std of each emebedding
def bayesian_predictions(model, X, estimators=10, device='cuda'):
    
    # The train in the evaluation setting is intended as it enables random behaviour, which is needed for the uncertanty computation
    model.train()

    with torch.no_grad():
     
        views = torch.cat([augment_image(X) for _ in range(estimators)], dim=0).to(device)  
        out = model(views)  

    B = X.size(0)
    out = out.view(estimators, B, -1)

    std = out.std(dim=0).mean(dim=1)  
    return std


# Predict the next 1000 samples, can reduce size we consider for speed
def predict_next(model, dataset, samples=20, k=4000, subset_ratio=0.7, used_indices=None):
    # Determine unused indices to prevent duplicates
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

    # Class-balanced sampling to not sample out-of-distribution and activate B4B
    labels = np.array(dataset.labels)
    label_counts = np.bincount(labels)
    label_probs = label_counts / label_counts.sum()
    topk_labels = labels[topk_indices]
    weights = label_probs[topk_labels]
    weights /= weights.sum()

    sampled_indices = np.random.choice(topk_indices, size=1000, replace=False, p=weights)
    return sampled_indices



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

    pbar = tqdm(range(epochs), leave=False)
    for epoch in pbar:
        model.train()
        total_loss = 0
        count = 0

        # Weights for balancing the loss functions
        alpha = (epoch / epochs) * 0.5

        for imgs, embeddings in loader:
            imgs, embeddings = imgs.cuda(), embeddings.cuda()
            
            # Augment images, stolen embedding used a different augmentation
            view_1 = augment_image(imgs).cuda()
            view_2 = augment_image(imgs).cuda()
           
            pred_embeddings_1 = model(view_1)
            pred_embeddings_2 = model(view_2)

            # Embeddings should be close to the victim and other augmentations of the same image
            mse_loss = criterion(pred_embeddings_1, embeddings)
            cos_loss = F.cosine_embedding_loss(pred_embeddings_1, embeddings, torch.ones(pred_embeddings_1.size(0)).cuda())
            cos_loss_2 = F.cosine_embedding_loss(pred_embeddings_1, pred_embeddings_2, torch.ones(pred_embeddings_1.size(0)).cuda())
         
            # At the start only use mse, if the model has learned slowly introduce the other two as well
            weighted_mse = (1 - alpha) * mse_loss
            weighted_cos = alpha * cos_loss
            weighted_cos_2 = alpha/2 * cos_loss_2
          
            loss = weighted_mse + weighted_cos + weighted_cos_2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            count += imgs.size(0)

        avg_loss = total_loss / count
     
        epoch_losses.append(avg_loss)

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
