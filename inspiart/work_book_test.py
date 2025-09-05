import os
import pandas as pd
from transformers import ViTImageProcessor, ViTModel, ViTImageProcessorFast
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.data_loaders import ImageLoader


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


image_folder = '/Users/shogun/code/gwen-m97/raw_data/sample1000'

images = [img for img in os.listdir(image_folder) if img.endswith('.jpg')]

image_loader = ImageLoader()

image_embbeding_function = GoogleVITHuge224Embedding()

chroma_client = chromadb.PersistentClient(path='/raw_data/google_vit_sample1000_db')

images_db = chroma_client.get_or_create_collection(name="google_vit_sample1000_collection", embedding_function=image_embbeding_function, data_loader=image_loader)

print("START")

for image in images:

    image_path = os.path.join(image_folder, image)

    print(image_path)

    image_pil = Image.open(image_path)

    meta_data_row = sample_data.query("file_name == {image}")

    metadata=[{'image_path' : image_path,
                    'artist': meta_data_row['artist'],
                    'style' : meta_data_row['style'],
                    'movement' : meta_data_row['movement'],
                    'url': meta_data_row['url'],
                    'img' : meta_data_row['img'],
                    'file_name' : meta_data_row['file_name'],
                    'genre_list' : meta_data_row['genre_list']}]

    images_db.add(
        ids = [image],
        uris = [image_path],
        metadatas=[metadata]
    )

print("FINISH")
