import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
import PIL.Image as Image

import numpy as np
import io

from sentence_transformers import SentenceTransformer, util
import os
from dotenv import load_dotenv
import torch
#import pandas as pd
import numpy as np
from PIL import Image
#import requests
#import matplotlib.pyplot as plt

import chromadb
#from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.data_loaders import ImageLoader

import json

app = FastAPI()

# # Allow all requests (optional, good for development purposes)
app.add_middleware(
     CORSMiddleware,
     allow_origins=["*"],  # Allows all origins
     allow_credentials=True,
     allow_methods=["*"],  # Allows all methods
     allow_headers=["*"],  # Allows all headers
 )

@app.get("/")
def index():
    return {"status": "ok CLIP"}

@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):

    #get the image from the POST request

    contents = img.file.read()

    working_image = Image.open(io.BytesIO(contents))

    #instantiate the image loader that ChromaDB uses to load pictures

    image_loader = ImageLoader()

    #connect to the database

    chroma_client = chromadb.CloudClient(
        api_key='ck-H5bhqzQ2aYVxtub2XUJNrJ2QmA3GApHDg1XDvFMSDg3x',
        tenant='153ed66b-a40a-4fd7-a05f-b9ce150bafac',
        database='inspiart'
        )

    #connect to the correct collection

    images_db = chroma_client.get_or_create_collection(name="wikiart_1000images", data_loader=image_loader)

    #instantiate the model

    model = SentenceTransformer('clip-ViT-B-32')

    # Use the CLIP model to encode the image
    query_embedding = model.encode(working_image).tolist()

    #perform the query

    image_suggestions = images_db.query(
    query_embeddings=[query_embedding],
    include=['uris','metadatas'],
    n_results=5
)


    image_dict = {'image_1': image_suggestions['metadatas'][0][0]['img'],
                  'image_2': image_suggestions['metadatas'][0][1]['img'],
                  'image_3': image_suggestions['metadatas'][0][2]['img'],
                  'image_4': image_suggestions['metadatas'][0][3]['img'],
                  'image_5': image_suggestions['metadatas'][0][4]['img']
                  }

    return_json = json.dumps(image_dict, indent=4)

    os.remove(query_uris)

    return return_json
