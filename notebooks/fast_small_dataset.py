import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
import PIL.Image as Image

import numpy as np
import io
import transformers

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
#from chromadb.utils.data_loaders import ImageLoader

import json

app = FastAPI()
app.state.model = SentenceTransformer('clip-ViT-B-32')

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

    #connect to the database

    chroma_client = chromadb.CloudClient(
        api_key='ck-H5bhqzQ2aYVxtub2XUJNrJ2QmA3GApHDg1XDvFMSDg3x',
        tenant='153ed66b-a40a-4fd7-a05f-b9ce150bafac',
        database='inspiart'
        )

    #get or create a connection

    images_db = chroma_client.get_or_create_collection(name="wikiart_1000images")

    # Use the CLIP model to encode the image

    query_embedding = app.state.model.encode(working_image).tolist()

    #perform the query

    image_suggestions = images_db.query(
    query_embeddings=[query_embedding],
    include=['uris','metadatas'],
    n_results=5
    )

    #create a dictionary of the results

    image_dict = {'image_1': image_suggestions['metadatas'][0][0]['url'],
                  'image_2': image_suggestions['metadatas'][0][1]['url'],
                  'image_3': image_suggestions['metadatas'][0][2]['url'],
                  'image_4': image_suggestions['metadatas'][0][3]['url'],
                  'image_5': image_suggestions['metadatas'][0][4]['url']
                  }
    #create a json of the dictionary

    return_json = json.dumps(image_dict, indent=4)

    #return the dictionary

    return return_json
