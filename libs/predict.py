import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as transforms

def predict_test(model, config):
    all_id = []
    all_pred = []
    test_augmentation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) # Data augmentation for validation and inference

    model.eval()
    for img_file in os.listdir(config['TEST_DATA_DIR_PATH']):
        img = Image.open(config['TEST_DATA_DIR_PATH'] + '/' + img_file)
        
        img = test_augmentation(img)
        
        img_batch = img[np.newaxis, :] # Convert to (1, ...) to simulate batch_size=1
        img_tensor = torch.as_tensor(img_batch).float().contiguous()
        img_tensor = img_tensor.to(config['DEVICE'], dtype=torch.float32, memory_format=torch.channels_last)
        name = os.path.splitext(img_file)[0]
        with torch.no_grad():
            pred = model(img_tensor)
            
        pred = pred.argmax()
        all_id.append(name)
        all_pred.append(pred.cpu().numpy())
        
    return pd.DataFrame({'id': all_id, 'type': all_pred}).sort_values(by='id')
