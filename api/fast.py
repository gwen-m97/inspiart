import os
# TODO: update path to use os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import PIL.Image as Image

import numpy as np
import io

from sentence_transformers import SentenceTransformer

import chromadb

from keras.applications.xception import preprocess_input
from tensorflow.keras import models

app = FastAPI()
app.state.model = SentenceTransformer('clip-ViT-B-32')
app.state.model_keras = models.load_model("models/model_Xception_alldata_finetuned.keras")
app.state.chroma_client = chromadb.CloudClient(
        api_key=os.environ.get("CHROMA_API_KEY"),
        tenant=os.environ.get("CHROMA_TENANT"),
        database='inspiart'
        )

# Allow all requests (optional, good for development purposes)
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

    #get or create a connection

    images_db = app.state.chroma_client.get_or_create_collection(name="wikiart_115000images")

    # Use the CLIP model to encode the image

    query_embedding = app.state.model.encode(working_image, device="cpu").tolist()

    #perform the query

    image_suggestions = images_db.query(
    query_embeddings=[query_embedding],
    include=['uris','metadatas', 'distances'],
    n_results=6
    )

    # Search if the image is matches with the first result as an output:
    distances = image_suggestions['distances'][0]

    is_exact_match = distances[0] < 10

    #create a dictionary of the results
    image_dict = {}

    if is_exact_match:
        for i in range(1,6) :
            key=f"image_{i-1}"
            image_dict[key] = {
                "img_url" : image_suggestions['metadatas'][0][i]['img_url'],
                "artist" : image_suggestions['metadatas'][0][i]['artist'],
                "file_name" : image_suggestions['metadatas'][0][i]['file_name'],
                "style" : image_suggestions['metadatas'][0][i]['style']
                }
    else :
        for i in range(5) :
            key=f"image_{i}"
            image_dict[key] = {
                "img_url" : image_suggestions['metadatas'][0][i]['img_url'],
                "artist" : image_suggestions['metadatas'][0][i]['artist'],
                "file_name" : image_suggestions['metadatas'][0][i]['file_name'],
                "style" : image_suggestions['metadatas'][0][i]['style']
                }

    final_dict = {"style_predicted" : None, "images" : image_dict}

    #return the dictionary

    return final_dict



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

    #get or create a connection

    images_db = app.state.chroma_client.get_or_create_collection(name="wikiart_115000images")

    # Use the CLIP model to encode the image

    query_embedding = app.state.model.encode(working_image, device="cpu").tolist()

    #perform the query

    image_suggestions = images_db.query(
    query_embeddings=[query_embedding],
    include=['uris','metadatas', 'distances'],
    n_results=6,
    where={"style": style_predicted}
    )

    # Search if the image is matches with the first result as an output:
    distances = image_suggestions['distances'][0]

    is_exact_match = distances[0] < 10

    #create a dictionary of the results

    image_dict = {}

    if is_exact_match:
        for i in range(1,6) :
            key=f"image_{i-1}"
            image_dict[key] = {
                "img_url" : image_suggestions['metadatas'][0][i]['img_url'],
                "artist" : image_suggestions['metadatas'][0][i]['artist'],
                "file_name" : image_suggestions['metadatas'][0][i]['file_name'],
                "style" : image_suggestions['metadatas'][0][i]['style']
                }
    else :
        for i in range(5) :
            key=f"image_{i}"
            image_dict[key] = {
                "img_url" : image_suggestions['metadatas'][0][i]['img_url'],
                "artist" : image_suggestions['metadatas'][0][i]['artist'],
                "file_name" : image_suggestions['metadatas'][0][i]['file_name'],
                "style" : image_suggestions['metadatas'][0][i]['style']
                }

    final_dict = {"style_predicted" : style_predicted, "images" : image_dict}

    #return the dictionary

    return final_dict




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

    #get or create a connection

    images_db = app.state.chroma_client.get_or_create_collection(name="wikiart_115000images")

    # Use the CLIP model to encode the image

    query_embedding = app.state.model.encode(working_image, device="cpu").tolist()

    #perform the query

    image_suggestions = images_db.query(
    query_embeddings=[query_embedding],
    include=['uris','metadatas', 'distances'],
    n_results=6,
    where={"style": {"$ne": style_predicted}}
    )

   # Search if the image is matches with the first result as an output:
    distances = image_suggestions['distances'][0]

    is_exact_match = distances[0] < 10

    #create a dictionary of the results

    image_dict = {}

    if is_exact_match:
        for i in range(1,6) :
            key=f"image_{i-1}"
            image_dict[key] = {
                "img_url" : image_suggestions['metadatas'][0][i]['img_url'],
                "artist" : image_suggestions['metadatas'][0][i]['artist'],
                "file_name" : image_suggestions['metadatas'][0][i]['file_name'],
                "style" : image_suggestions['metadatas'][0][i]['style']
                }
    else :
        for i in range(5) :
            key=f"image_{i}"
            image_dict[key] = {
                "img_url" : image_suggestions['metadatas'][0][i]['img_url'],
                "artist" : image_suggestions['metadatas'][0][i]['artist'],
                "file_name" : image_suggestions['metadatas'][0][i]['file_name'],
                "style" : image_suggestions['metadatas'][0][i]['style']
                }

    final_dict = {"style_predicted" : style_predicted, "images" : image_dict}

    #return the dictionary

    return final_dict


@app.post('/samepainting_search')
async def receive_image(img: UploadFile=File(...)):

    #get the image from the POST request

    contents = img.file.read()

    working_image = Image.open(io.BytesIO(contents))

    #get or create a connection

    images_db = app.state.chroma_client.get_or_create_collection(name="wikiart_115000images")

    # Use the CLIP model to encode the image

    query_embedding = app.state.model.encode(working_image, device="cpu").tolist()

    #perform the query

    image_suggestions = images_db.query(
    query_embeddings=[query_embedding],
    include=['uris','metadatas', 'distances'],
    n_results=1
    )

    # Search if the image is matches with the first result as an output:
    distances = image_suggestions['distances'][0]

    is_exact_match = distances[0] < 10

    # if it matches, return its infos, else return we don't know the image
    if is_exact_match:
        image_dict = {
            "artist" : image_suggestions['metadatas'][0][0]['artist'],
            "file_name" : image_suggestions['metadatas'][0][0]['file_name']
            }
    else :
        image_dict = {
            "artist" : "Unknown artist",
            "file_name" : "Unknown artwork"
            }

    return image_dict
