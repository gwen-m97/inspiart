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

data='/kaggle/input/wikiart-all-artpieces/wikiart_art_pieces.csv'

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

embeddings = []
for path in tqdm(df['filepath']):
    try:
        emb = extract_embedding(path)
        embeddings.append(emb)
    except Exception as e:
        print(f"Error with {path}: {e}")
        embeddings.append(np.zeros(2048))  # placeholder

df['embedding'] = embeddings


# Compute similarity between all pairs
embedding_matrix = np.stack(df['embedding'].values)
similarity_matrix = cosine_similarity(embedding_matrix)

# Find most similar images to the first image
top_k = 5
similar_indices = np.argsort(-similarity_matrix[0])[:top_k+1]  # +1 to skip self
print("Most similar images to:", df.iloc[0]['filepath'])
for i in similar_indices[1:]:  # skip the image itself
    print(df.iloc[i]['filepath'], "→ similarity:", similarity_matrix[0][i])


tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(embedding_matrix)

plt.figure(figsize=(10, 8))
plt.scatter(reduced[:, 0], reduced[:, 1], c=pd.factorize(df['label'])[0], cmap='tab10')
plt.title("t-SNE Visualization of Image Embeddings (by Style)")
plt.show()
