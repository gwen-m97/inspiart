from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import os
import pandas as pd
import chromadb
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="image_tensors",metadata={"hnsw:space": "cosine"} )

#url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
#image = Image.open(requests.get(url, stream=True).raw)
#display(image)

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-huge-patch14-224-in21k')
model = ViTModel.from_pretrained('google/vit-huge-patch14-224-in21k')

image_vector_df = pd.DataFrame([['fred','john']], columns=['image', 'image_vector'])
image_folder = '/Users/shogun/code/gwen-m97/raw_data/Pop_Art'

count=0
batch_size= 10

images = [img for img in sorted(os.listdir(image_folder)) if img.endswith('.jpg')]
image_vector_df_path = ''.join((image_folder, "\image_vector_df_csv.csv"))

for image in images:
    image_path = os.path.join(image_folder, image)
    image_pil = Image.open(image_path)
    #image_pil.show()
    inputs = feature_extractor(images=image_pil, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    image_vector_df = pd.concat([image_vector_df, pd.DataFrame([[image, last_hidden_states]], columns=image_vector_df.columns)],ignore_index=True)




    #count += 1
    #if count >= batch_size:
    #    count = 0
    #    image_vector_df.to_csv(image_vector_df_path)
    #    print(batch_size)
