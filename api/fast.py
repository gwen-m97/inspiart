from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

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
    ### Receiving and decoding the image
    contents = await img.read()

    print('IN POST')

    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

    ### Do cool stuff with your image.... For example face detection

    #chroma_client = chromadb.PersistentClient(path='/Users/shogun/code/gwen-m97/inspiart/raw_data/my_vectordb')
    #image_loader = ImageLoader()

    #image_embbeding_function = GoogleVITHuge224Embedding()
    #images_db = chroma_client.get_or_create_collection(name="images_db", embedding_function=image_embbeding_function, data_loader=image_loader)
    #query_uris = '/Users/shogun/code/gwen-m97/raw_data/test_images/Two_Young_Girls_at_the_Piano_MET_rl1975.1.201.R.jpg'

    #image_suggestions = images_db.query(

    #query_uris=[query_uris]
#)

#    print(image_suggestions)

    #annotated_img = cv2_img

    ### Encoding and responding with the image
    im = cv2.imencode('.jpg', cv2_img)[1] # extension depends on which format is sent from Streamlit
    #return image_suggestions
    return Response(content=im.tobytes(), media_type="image/jpg")
