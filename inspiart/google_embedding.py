from transformers import ViTImageProcessor, ViTModel, ViTImageProcessorFast


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
