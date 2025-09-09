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

#imports for the style model

from keras.applications.xception import preprocess_input
import tensorflow as tf

app = FastAPI()
app.state.model = SentenceTransformer('clip-ViT-B-32')
app.state.model_keras = keras.models.load_model("../models/model_Xception_alldata_finetuned.keras")


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

    images_db = chroma_client.get_or_create_collection(name="wikiart_115000images")

    # Use the CLIP model to encode the image

    query_embedding = app.state.model.encode(working_image).tolist()

    #perform the query

    image_suggestions = images_db.query(
    query_embeddings=[query_embedding],
    include=['uris','metadatas'],
    n_results=5
    )
                  }
    #create a json of the dictionary

    image_dict = {}

    for i in range(5) :
        key=f"image_{i}"
        image_dict[key] = {
            "img_url" : image_suggestions['metadatas'][0][i]['img_url'],
            "artist" : image_suggestions['metadatas'][0][i]['artist'],
            "file_name" : image_suggestions['metadatas'][0][i]['file_name'],
            "style" : image_suggestions['metadatas'][0][i]['style']
            }

    final_dict = {"style_predicted" : None, "images" : image_dict}

    return_json = json.dumps(final_dict)

    #return the dictionary

    return return_json



@app.post('/upload_same_style')
async def receive_image(img: UploadFile=File(...)):

    #get the image from the POST request

    contents = img.file.read()

    working_image = Image.open(io.BytesIO(contents))

    #GET STYLE

    #styles constant

    LIST_STYLES = ['Abstract Art', 'Abstract Expressionism', 'Academicism', 'Art Deco', 'Art Informel', 'Art Nouveau (Modern)', 'Biedermeier', 'Color Field Painting', 'Conceptual Art', 'Concretism', 'Contemporary', 'Contemporary Realism', 'Cubism', 'Dada', 'Divisionism', 'Expressionism', 'Fantastic Realism', 'Fauvism', 'Figurative Expressionism', 'Futurism', 'Hard Edge Painting', 'Hyper-Realism', 'Impressionism', 'Kitsch', 'Luminism', 'Lyrical Abstraction', 'Magic Realism', 'Metaphysical art', 'Minimalism', 'Native Art', 'Naturalism', 'Naïve Art (Primitivism)', 'Neo-Dada', 'Neo-Expressionism', 'Neo-Impressionism', 'Neo-Pop Art', 'Neo-Romanticism', 'Neoclassicism', 'New European Painting', 'Op Art', 'Orientalism', 'Pop Art', 'Post-Impressionism', 'Post-Painterly Abstraction', 'Precisionism', 'Realism', 'Regionalism', 'Romanticism', 'Social Realism', 'Socialist Realism', 'Surrealism', 'Symbolism', 'Synthetic Cubism', 'Tachisme', 'Tonalism', 'Transavantgarde']

    #going to find the style

    #PREPROCESSING

    img = working_image.convert('RGB')
    img_resized = img.resize((224, 224), Image.BICUBIC)
    img_array = np.array(img_resized)
    img_preprocessed = preprocess_input(img_array)
    img_batch = np.expand_dims(img_preprocessed, axis=0)  # shape (1,224,224,3)

    #PREDICTION

    preds = app.state.model_keras.predict(img_batch)
    pred_indice = preds.argmax(axis=1)[0] #Take the number
    style_predicted = LIST_STYLES[pred_indice]

    #GET IMAGES THAT MATCH WITH STYLE AND IMAGE

    #connect to the database

    chroma_client = chromadb.CloudClient(
        api_key='ck-H5bhqzQ2aYVxtub2XUJNrJ2QmA3GApHDg1XDvFMSDg3x',
        tenant='153ed66b-a40a-4fd7-a05f-b9ce150bafac',
        database='inspiart'
        )

    #get or create a connection

    images_db = chroma_client.get_or_create_collection(name="wikiart_115000images")

    # Use the CLIP model to encode the image

    query_embedding = app.state.model.encode(working_image).tolist()

    #perform the query

    image_suggestions = images_db.query(
    query_embeddings=[query_embedding],
    include=['uris','metadatas'],
    n_results=5,
    where={"style": style_predicted}
    )

    #create a dictionary of the results

    image_dict = {}

    for i in range(5) :
        key=f"image_{i}"
        image_dict[key] = {
            "img_url" : image_suggestions['metadatas'][0][i]['img_url'],
            "artist" : image_suggestions['metadatas'][0][i]['artist'],
            "file_name" : image_suggestions['metadatas'][0][i]['file_name'],
            "style" : image_suggestions['metadatas'][0][i]['style']
            }

    final_dict = {"style_predicted" : style_predicted, "images" : image_dict}

    #create a json of the dictionary

    return_json = json.dumps(final_dict)

    #return the dictionary

    return return_json



@app.post('/upload_other_style')
async def receive_image(img: UploadFile=File(...)):

    #get the image from the POST request

    contents = img.file.read()

    working_image = Image.open(io.BytesIO(contents))

    #GET STYLE

    #styles constant

    LIST_STYLES = ['Abstract Art', 'Abstract Expressionism', 'Academicism', 'Art Deco', 'Art Informel', 'Art Nouveau (Modern)', 'Biedermeier', 'Color Field Painting', 'Conceptual Art', 'Concretism', 'Contemporary', 'Contemporary Realism', 'Cubism', 'Dada', 'Divisionism', 'Expressionism', 'Fantastic Realism', 'Fauvism', 'Figurative Expressionism', 'Futurism', 'Hard Edge Painting', 'Hyper-Realism', 'Impressionism', 'Kitsch', 'Luminism', 'Lyrical Abstraction', 'Magic Realism', 'Metaphysical art', 'Minimalism', 'Native Art', 'Naturalism', 'Naïve Art (Primitivism)', 'Neo-Dada', 'Neo-Expressionism', 'Neo-Impressionism', 'Neo-Pop Art', 'Neo-Romanticism', 'Neoclassicism', 'New European Painting', 'Op Art', 'Orientalism', 'Pop Art', 'Post-Impressionism', 'Post-Painterly Abstraction', 'Precisionism', 'Realism', 'Regionalism', 'Romanticism', 'Social Realism', 'Socialist Realism', 'Surrealism', 'Symbolism', 'Synthetic Cubism', 'Tachisme', 'Tonalism', 'Transavantgarde']

    #going to find the style

    #PREPROCESSING

    img = working_image.convert('RGB')
    img_resized = img.resize((224, 224), Image.BICUBIC)
    img_array = np.array(img_resized)
    img_preprocessed = preprocess_input(img_array)
    img_batch = np.expand_dims(img_preprocessed, axis=0)  # shape (1,224,224,3)

    #PREDICTION

    preds = app.state.model_keras.predict(img_batch)
    pred_indice = preds.argmax(axis=1)[0] #Take the number
    style_predicted = LIST_STYLES[pred_indice]

    #GET IMAGES THAT MATCH WITH STYLE AND IMAGE

    #connect to the database

    chroma_client = chromadb.CloudClient(
        api_key='ck-H5bhqzQ2aYVxtub2XUJNrJ2QmA3GApHDg1XDvFMSDg3x',
        tenant='153ed66b-a40a-4fd7-a05f-b9ce150bafac',
        database='inspiart'
        )

    #get or create a connection

    images_db = chroma_client.get_or_create_collection(name="wikiart_115000images")

    # Use the CLIP model to encode the image

    query_embedding = app.state.model.encode(working_image).tolist()

    #perform the query

    image_suggestions = images_db.query(
    query_embeddings=[query_embedding],
    include=['uris','metadatas'],
    n_results=5,
    where={"style": {"$ne": style_predicted}}
    )

    #create a dictionary of the results

    image_dict = {}

    for i in range(5) :
        key=f"image_{i}"
        image_dict[key] = {
            "img_url" : image_suggestions['metadatas'][0][i]['img_url'],
            "artist" : image_suggestions['metadatas'][0][i]['artist'],
            "file_name" : image_suggestions['metadatas'][0][i]['file_name'],
            "style" : image_suggestions['metadatas'][0][i]['style']
            }

    final_dict = {"style_predicted" : style_predicted, "images" : image_dict}

    #create a json of the dictionary

    return_json = json.dumps(final_dict)

    #return the dictionary

    return return_json
