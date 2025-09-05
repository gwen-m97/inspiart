import streamlit as st
import numpy as np
from PIL import Image
import numpy as np
import requests
from PIL import Image
from io import BytesIO



# call two models

# supervised_model = tf.keras.models.load_model("supervised_resnet18.h5")

# Load unsupervised embedding model

# URL dictonary (replace with the original one)
url_list = [
    "https://picsum.photos/id/237/300/300",
    "https://picsum.photos/id/238/300/300",
    "https://picsum.photos/id/239/300/300"
]



# unsupervised_model = tf.keras.models.load_model("unsupervised_split.h5")

# image_embeddings = np.load("embeddings.npy")   # shape: (N, D)
# image_paths = np.load("image_paths.npy", allow_pickle=True)


# Display logo from relative path

# At the top of app.py
st.image("../raw_data/logo.png", width=200)
st.write("Upload an image to predict style and find similar artworks.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)


    #  Predict Style
    if st.button("Predict Style"):
        # preds = request.post("http://localhost:8000/predict_style", files={"file": uploaded_file}).json()
        # class_id = np.argmax(preds, axis=1)[0]
        # st.write(f"**Predicted Style:** {style_names[class_id]}")
        st.write("Style: (placeholder, plug in your supervised model)")

    # Find Similar Images
    if st.button("Find Similar Images"):
        # embedding = unsupervised_model.predict(input_tensor)
        # distances, indices = index.search(embedding.astype(np.float32), 5)

        st.subheader("Top 5 Similar Images")
        cols = st.columns(5)
        for i in range(5):
            # sim_img = Image.open(image_paths[indices[0][i]])
            sim_img = img  # placeholder
            cols[i].image(sim_img, use_column_width=True)


##streamit run frontend_prototype.py
