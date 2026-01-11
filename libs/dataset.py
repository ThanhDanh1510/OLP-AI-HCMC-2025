import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms

FOLDER_TO_LABEL = {
    'bào ngư xám + trắng': 1,
    'Đùi gà Baby (cắt ngắn)': 2,
    'linh chi trắng': 3,
    'nấm mỡ': 0
}

class CustomDataset(Dataset):
    def __init__(self, data_path, is_valid = False):
        self.data_path = data_path
        self.is_valid = is_valid
        
        self.train_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomSolarize(threshold=20.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) # Data augmentation for training
        self.val_augmentation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) # Data augmentation for validation and inference

        self.data_path = data_path
        
        self.all_folders = os.listdir(data_path)
        self.real_classes = [2, 0, 3, 1]
        
        self.imgs_path = []
        self.labels = []
        for folder_name in self.all_folders:
            class_folder = os.path.join(self.data_path, folder_name)
            for img_file in os.listdir(class_folder):
                img_full_path = os.path.join(class_folder, img_file)
                self.imgs_path.append(img_full_path)
                self.labels.append(FOLDER_TO_LABEL[folder_name])
        
        # Convert lists to numpy arrays for convenience
        self.imgs_path = np.array(self.imgs_path)
        self.labels = np.array(self.labels)
        
    def __len__(self):
        return len(self.imgs_path)

    def transform(self, img):
        if not self.is_valid:
            img = self.train_augmentation(img)
        else:
            img = self.val_augmentation(img)
        return img

    # Get item for training
    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path)
        img = self.transform(img)
        
        return {
            'images': torch.as_tensor(img).float().contiguous(),
            'labels': torch.as_tensor(label).long().contiguous()
        }
    
def get_dataloader(data_path, batch_size, num_workers, is_valid=False):
    dataset = CustomDataset(data_path, is_valid=is_valid)
    train_idx, valid_idx = train_test_split(np.arange(len(dataset)),
                                                test_size=0.2,
                                                shuffle=True,
                                                stratify=dataset.labels)

    train_subset = Subset(dataset, train_idx)
    validation_subset = Subset(dataset, valid_idx)

    train_dataloader = DataLoader(train_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(validation_subset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_dataloader, val_dataloader