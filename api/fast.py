import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
import PIL.Image as Image

import numpy as np
import cv2
import io

from transformers import ViTImageProcessor, ViTModel, ViTImageProcessorFast

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.data_loaders import ImageLoader

#from face_rec.face_detection import annotate_face

app = FastAPI()

# # Allow all requests (optional, good for development purposes)
app.add_middleware(
     CORSMiddleware,
     allow_origins=["*"],  # Allows all origins
     allow_credentials=True,
     allow_methods=["*"],  # Allows all methods
     allow_headers=["*"],  # Allows all headers
 )

class GoogleVITHuge224Embedding(EmbeddingFunction):

    '''
    A class to provide custom embeddings to a ChromaDB database
    embedding images using the Google vit-huge-patch14-224-in21k
    the class returns an embedding as a numpy array
    '''


    def __call__(self, input: Documents) -> Embeddings:

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

@app.get("/")
def index():
    return {"status": "ok"}

@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):
    print(type(img))
    ### Receiving and decoding the image
    #contents = img.file.read()
    contents = img.file.read()
    print(type(contents))
    #nparr = np.fromstring(contents, np.uint8)
    #cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

    ### Do cool stuff with your image.... For example face detection

    # save a local copy of the file to get the uri

    working_image = Image.open(io.BytesIO(contents))

    directory = '/Users/shogun/code/gwen-m97/inspiart/api/working_directory/working_image.jpg'

    working_image.save(directory)

    #instantiate the image loader that ChromaDB uses to load pictures

    image_loader = ImageLoader()

#intantiate the custom embedding function

    image_embbeding_function = GoogleVITHuge224Embedding()

#connect to the database

    chroma_client = chromadb.PersistentClient(path='/Users/shogun/code/gwen-m97/inspiart/models/google_vit_sample1000_db')

#connect to the correct collection

    images_db = chroma_client.get_or_create_collection(name="google_vit_sample1000_collection", embedding_function=image_embbeding_function, data_loader=image_loader)

#test picture string

    #query_uris = '/Users/shogun/code/gwen-m97/raw_data/test_images/Two_Young_Girls_at_the_Piano_MET_rl1975.1.201.R.jpg'
    #query_uris = '/Users/shogun/code/gwen-m97/raw_data/test_images/Piet_Mondriaan,_1942_-_New_York_City_I.jpg'
    #query_uris = '/Users/shogun/code/gwen-m97/raw_data/test_images/Paul_Cézanne_-_The_Basket_of_Apples_-_1926.252_-_Art_Institute_of_Chicago.jpg'
    #query_uris = '/Users/shogun/code/gwen-m97/raw_data/test_images/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg'
    query_uris = '/Users/shogun/code/gwen-m97/inspiart/api/working_directory/working_image.jpg'

#perform the query

    image_suggestions = images_db.query(
    query_uris=[query_uris], include=['uris','metadatas'], n_results=5
)

#url=f"{image_suggestions['metadatas'][0][9]['img']}"
    image_dict = {'image_1': image_suggestions['metadatas'][0][0]['img'],
                  'image_2': image_suggestions['metadatas'][0][1]['img'],
                  'image_3': image_suggestions['metadatas'][0][2]['img'],
                  'image_4': image_suggestions['metadatas'][0][3]['img'],
                  'image_5': image_suggestions['metadatas'][0][4]['img']
                  }

    #annotated_img = cv2_img

    ### Encoding and responding with the image
    #im = cv2.imencode('.jpg', image)[1] # extension depends on which format is sent from Streamlit
    #return image_suggestions
    #return Response(image_suggestions['metadatas'][0]['img'])

    os.remove(query_uris)

    return image_dict
