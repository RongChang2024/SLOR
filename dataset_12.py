import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class CrohnsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.classes = sorted(os.listdir(data_dir))
        
        self.person_data = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            for person_name in os.listdir(class_dir):
                person_dir = os.path.join(class_dir, person_name)
                
                info_file_path = os.path.join(person_dir, 'info.txt')
                if not os.path.exists(info_file_path):
                    continue
                
                with open(info_file_path, 'r') as f:
                    info_content = f.read().strip()
                
                person_images = []
                for image_name in os.listdir(person_dir):
                    if image_name.endswith(('.png', 'jpg')):
                        image_path = os.path.join(person_dir, image_name)
                        person_images.append(image_path)
                        
                if len(person_images) == 12:
                    self.person_data.append((person_images, class_idx, person_name, info_content))
        
    def __len__(self):
        return len(self.person_data)

    def __getitem__(self, index):
        person_images, label, person_name, info_content = self.person_data[index]


        images = torch.zeros(12, 3, 224, 224)
        for i, image_path in enumerate(person_images):
            image = Image.open(image_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            images[i] = image
        return images, label, person_name, info_content
