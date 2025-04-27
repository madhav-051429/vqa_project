import os
from torchvision import transforms

IMAGE_SIZE = 224

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                         (0.26862954, 0.26130258, 0.27577711))
])

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
