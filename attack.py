import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle



from src.utils import *
from src.dataset import *
from src.model import * 
from src.api import *

# Data transform was given
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2980, 0.2962, 0.2987],
                         std=[0.2886, 0.2875, 0.2889]),

])




def attack(dataset, rounds, device='cuda'):
   
    selected_indices = np.array([])
    training_data = EmbeddingDataset(transform=transform)
    used_indices = []
    
    # We train one model on the entire data
    surrogate_model = SmallResNet18(output_dim=1024, pretrained=True)
    
    for i in range(rounds):
        
        # At the start select random indices
        if selected_indices.size == 0:
            selected_indices = np.random.permutation(1000)
        else:
            
            # Every 5 rounds instead retrain on the full dataset to prevent "forget", else only train on the newly added ones
            if i % 5 == 0:
                loader = DataLoader(training_data, batch_size=128, shuffle=True)
            else:   
                loader = DataLoader(current_data, batch_size=128, shuffle=True)

            if len(dataset) - len(used_indices) < 1000:
               loader = DataLoader(training_data, batch_size=128, shuffle=True)
               surrogate_model  = train(surrogate_model, loader,50, i)
               
               break
            
            surrogate_model  = train(surrogate_model, loader,20, i)
            
            # Perform MC-Dropout to get the new indices we want to query
            selected_indices = predict_next(surrogate_model, dataset, used_indices=used_indices,samples=15)
            used_indices.extend(selected_indices.tolist())
        
        
        # We then augment and send them to the api to encode
        images = [augment_image_single(dataset.imgs[idx]) for idx in selected_indices]  
        embeddings = model_stealing(images, port=PORT, TOKEN=TOKEN)  
        
        # Embeddings are saved for potential further analysis
        with open(f'api_out/embeddings{i}.pickle', 'wb') as handle:
            pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Extend the datasets with the new data
        current_data = EmbeddingDataset(transform=transform)
        current_data.extend(images,embeddings)
        training_data.extend(images,embeddings)


# Load the dataset
dataset = torch.load("data/ModelStealingPub.pt",weights_only=False)
dataset.transform = transform

# Change the labels to be 0,1,2,...,n
original_labels = np.array(dataset.labels)
unique_labels = np.unique(original_labels)
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=0)}
new_labels = np.array([label_mapping[label] for label in original_labels])
dataset.labels = new_labels.tolist()


TOKEN = "08392413"
SEED = "40069910"
PORT = "9817"


# Code to request a new api, attack and upload the model

# new_api(TOKEN)
# attack(dataset, rounds=400)  
upload("model_out/surrogate_model9.pth",TOKEN,SEED) 

