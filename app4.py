import streamlit as st
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor

# Directory path to the saved model on Google Drive
model_directory = 'kusumakargit/openin/"

model = VisionEncoderDecoderModel.from_pretrained("model_directory")

# create the Streamlit app
def app():
    st.title('sample')
    st.write('Upload an image and see the predictions of two CNN models!')

    # create file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # check if file has been uploaded
    if uploaded_file is not None:
        # load the image
        image = Image.open(uploaded_file)

        generated_caption = tokenizer.decode(model.generate(feature_extractor(img, return_tensors="pt").pixel_values.to("cpu"))[0])
        sentence = generated_caption
        text_to_remove = "<|endoftext|>"
        generated_caption = sentence.replace(text_to_remove, "")
      

        # display the image
        st.image(image, caption=generated_caption, use_column_width=True)
        st.write(generated_caption)

# run the app
if __name__ == '__main__':
    app()
