import os
import datetime

from transformers import ViTImageProcessor, ViTModel, ViTImageProcessorFast
from PIL import Image
import requests

import pandas as pd
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
#from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

class GoogleVITHuge224Embedding(EmbeddingFunction):


    def __call__(self, input: Documents) -> Embeddings:

        from transformers import ViTImageProcessor, ViTModel, ViTImageProcessorFast

        #Instantiate the image. Convert it to 244 x 244 and normalise RGB between 0 and 1 witha mean of 0.5 for each channel

        self.feature_extractor = ViTImageProcessorFast.from_pretrained('google/vit-huge-patch14-224-in21k')

        #Instantiate the Google ViT with pretrained weights

        self.model = ViTModel.from_pretrained('google/vit-huge-patch14-224-in21k')#Preprocess the data

        inputs = self.feature_extractor(images=input, return_tensors="pt")

        #Embedd the data

        outputs = self.model(**inputs)

        #Convert the embedding to a Numpy array and take the first vector of the Transformer state

        embeddings = outputs.last_hidden_state.data.numpy()[0,0]

        #return the embedding

        return embeddings



#url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
#image = Image.open(requests.get(url, stream=True).raw)
#display(image)

feature_extractor = ViTImageProcessorFast.from_pretrained('google/vit-huge-patch14-224-in21k')
model = ViTModel.from_pretrained('google/vit-huge-patch14-224-in21k')


image_vector_df = pd.DataFrame([], columns=['image', 'image_vector'])
image_numpy_df = pd.DataFrame([])
image_df = pd.DataFrame([])


image_folder = '/Users/shogun/code/gwen-m97/raw_data/test_images'

#count=0
#batch_size= 10
#batch_multiplier = 1

log_file_name = f"log_file_{datetime.datetime.now()}.txt\n"

images = [img for img in sorted(os.listdir(image_folder)) if img.endswith('.jpg')]
image_vector_df_path = ''.join((image_folder, "/test_images_vector_df_csv.csv"))

chroma_client = chromadb.Client()
image_collection = chroma_client.create_collection(name="image_collection", embedding_function=GoogleVITHuge224Embedding())

print("START")

for image in images:
    image_path = os.path.join(image_folder, image)
    image_pil = Image.open(image_path)
    #if not image_pil.mode == 'RGB':
    #    image_pil = image_pil.convert('RGB')
    #image_pil.show()
    inputs = feature_extractor(images=image_pil, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state.data.numpy()[0,0]

    #import ipdb; ipdb.set_trace()

    last_hidden_states = last_hidden_states.squeeze()

    image_numpy_df = pd.concat([image_numpy_df, pd.DataFrame([last_hidden_states])])

    print(image_numpy_df)

    image_df = pd.concat([pd.DataFrame([image]), image_df], ignore_index=True)

    print(image_df)

    image_vector_df = pd.concat([image_vector_df, pd.DataFrame([[image, last_hidden_states]], columns=image_vector_df.columns)],ignore_index=True)

    print(image_vector_df)

    #if count >= batch_size:
    with open(log_file_name, "a") as log_file:
        log_file.write(f"{image} - {datetime.datetime.now()}\n")

    image_vector_df.to_csv(image_vector_df_path)

    break

print("FINISH")
    #print(batch_size * batch_multiplier)
    #count = 0
    #batch_multiplier += 1






    #count += 1
    #if count >= batch_size:
    #    count = 0
    #    image_vector_df.to_csv(image_vector_df_path)
    #    print(batch_size)
