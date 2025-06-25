# TML Assignment 2

## Files and Their Descriptions  

attack.py - Main code for the model stealing  
            attack: runs the attack for n rounds, each round sending 1000 new images to the api. Returns the stolen model at the end

api.py - Handles all the code which interacts with the api.  
        new_api: Requests a new api/model. Returns TOKEN and SEED.   
        model_stealing: Sends 1000 images to the api. Returns the embeddings.  
        upload: Uploads the current stolen model for evaluation. Returns L2 distance to original model.   
        embeddings_sim: Measures the cosine similarity between 2 different subsets. Returns the average cosine similarity.  

utils.py - This file includes different functions required to steal the model.  
           augment_image and augment_image_single: Augment batch of images and a single image according to AUGMENT and AUGMENT_SINGLE. Return the augmented version.  
           bayesian_predictions: Uses MC-Dropout to compute the uncertainty for each datapoint. Returns the mean std over the embedding.  
           predict_next: Uses bayesian_predictions and the class distribution to generate new candidates. Returns the next 1000 images to be sent to the api.  
           train: trains the model's output to be close to the real encoder's outputs. Returns the trained model.  

model.py - The predicted model architecture is stored here.  

data_sets.py - In here, the  Dataset classes are stored, one for the original data and one for the stolen embeddings together with the predicted embeddings.  

## Dependencies

The following Python packages and modules are required to run the scripts:

- torch  
- torchvision  
- requests  
- onnxruntime  
- tqdm  
- kornia  
- timm  
- scikit-learn

### Custom modules
- src.utils  
- src.dataset  
- src.model  
- src.api  
