![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Google Cloud](https://img.shields.io/badge/GCP-4285F4?logo=googlecloud&logoColor=white)



# 🎨 Inspiart: An AI companion to inspire your art journey

**Inspiart** is a machine learning–powered tool that recommends paintings similar to a given artwork. It can also detect its artistic style.
The project is available through an interactive Streamlit app.

![App Video demo short 1](assets/Short_demo_video_1.gif)


---


## Features
- 🔎 Image-based recommendation of similar artworks
- 🖼️ Automatic detection and classification of artistic styles
- 📊 Interactive web interface to explore the tool


![App Screenshot](assets/Demo-pic-2.png)

---

## Repository Structure
This project is split into two repositories:
- [inspiart](https://github.com/gwen-m97/inspiart): core code (data preprocessing, model training, recommendation engine, API)
- [inspiart-front](https://github.com/gwen-m97/inspiart-front): Streamlit app

```bash
└── inspiart
    ├── Dockerfile
    ├── setup.py
    ├── requirements.txt
    ├── Makefile
    ├── api
    │   └─ fast.py
    ├── models
    │   └─ model_Xception_alldata_finetuned.keras
    │   └─ model_clip
    └── inspiart
        └── dataset_prep.py
```

```bash
└── inspiart-front
    ├── app.py
    └── sources

```


---

## Technical Overview

### 📂 Dataset
- Used the **[WikiArt dataset](https://www.kaggle.com/datasets/simolopes/wikiart-all-artpieces)** from Kaggle
- Contains more than 170K artworks across different artistic styles
- Cleaning to only keep paintings after 1800. 114K images in the end.

### 🤖 Models

1. **CLIP (OpenAI)**
   - Extracted embeddings for all paintings in the dataset
   - Used these embeddings to perform **image similarity search**
   - Qualitatively good results, often retrieves artworks that are obviously related and with same content

2. **Custom Model (Xception-based)**
   - Fine-tuned Xception with additional dense layers for **artistic style classification**
   - Final accuracy achieved: **47%** on the test set (30% of the database)


### 🗄️ Data Storage
- Stored all embeddings in a **ChromaDB** vector database
- Enables fast similarity search across the artwork collection


### ⚙️ Backend
- Built an **API with FastAPI** to serve both:
  - Similarity search results (CLIP)
  - Style predictions (custom Xception model)
- Packaged into a **Docker image** for easier deployment


### ☁️ Deployment
- Deployed the backend on **Google Cloud Platform (GCP)**
- Frontend built with **Streamlit** for an interactive user interface


### 🖼️ Architecture Overview
![Architecture](assets/Architecture.png)


---


## Demo
How to use Inspiart ?
![App user path](assets/Demo-pic-1.png)

📹 Demo Video
[Watch full demo on YouTube](https://www.youtube.com/watch?v=uE1tRA1KkAE)

![App Video demo short 2](assets/Short_demo_video_2.gif)

Final presentation :
📄 [View the final presentation (PDF)](assets/Inspiart_final_presentation.pdf)



## Project context
Team project realized during the Data Science & AI Bootcamp at Le Wagon.
2 weeks with a team of 5 to develop a solution using data and AI.
