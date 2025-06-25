import numpy as np
import torch
import requests
import onnxruntime as ort
import json
import io
import sys
import base64
import time
from torchvision.transforms import ToPILImage

from src.utils import *
from src.model import *



# Given code to rqeuest a new api
def new_api(TOKEN):
    response = requests.get("http://34.122.51.94:9090" + "/stealing_launch", headers={"token": TOKEN})
    answer = response.json()

    print(answer)  # {"seed": "SEED", "port": PORT}

    # Save to file
    with open("data.txt", "w") as f:
        json.dump(answer, f)

    if 'detail' in answer:
        sys.exit(1)


# Given code to query the api
def model_stealing(images, port, TOKEN):
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

# Given code to upload the solution
def upload(model_path, TOKEN, SEED):
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




# We try to estimate how the embeddings are changed through B4B by comparing the cosine sim of two seperate sets of images
def embeddings_sim(dataset, PORT):
   

    time.sleep(60)
    selected = np.random.permutation(len(dataset))[:1000]

    to_pil = ToPILImage()
    
 
    images = [augment_image_single(to_pil(dataset[i][1])) for i in selected]  
    embeddings = torch.tensor(model_stealing(images, port=PORT))

    selected = np.random.permutation(len(dataset))[:1000]
    images = [augment_image_single(to_pil(dataset[i][1])) for i in selected] 
    time.sleep(60)
    
    embeddings2 = torch.tensor(model_stealing(images, port=PORT))

    time.sleep(60)
    

    similarities = F.cosine_similarity(embeddings, embeddings2, dim=1).cpu().numpy()
    return similarities.mean()

