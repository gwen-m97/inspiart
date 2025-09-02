import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, log_loss
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


image_extensions=('.jpg', '.jpeg', '.png')

filepath=[]
labels=[]

data='raw_data/data_sampling200_topstyles.csv'

for dirname, _, filenames in os.walk(data):
    for filename in filenames:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(dirname, filename)
            filepaths.append(full_path)
            labels.append(os.path.basename(dirname))


#dataframe
df= pd.DataFrame({
    'filepath': filepaths,
    'label': labels
})

# Load pretrained ResNet50
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove final layer
resnet.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(img_tensor).squeeze().cpu().numpy()  # shape: (2048,)
    return embedding


resnet = models.resnet50(pretrained=True) #isntantiating a ResNet-50 model with pre-trained weights
resnet.fc = nn.Identity()  # removes classification head

#classification head
class ClassificationHead(nn.Module):
    def __init__(self, input_dim=2048, num_classes=10):
        super(ClassificationHead, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10)  # final layer
        )

    def forward(self, x):
        return self.model(x)

# Complete model
class ResNetWithClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetWithClassifier, self).__init__()
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.fc = nn.Identity()
        self.classifier = ClassificationHead(2048, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
