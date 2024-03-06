from transformers import pipeline
import os
import requests
import json
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("HUGGING_FACE_API")


#img to text
def img2text(url):
    img_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = img_to_text(url)[0]['generated_text']

    print(text)
    return text


#text to story
#serverless implimentation 

def text_to_story(text):
    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    response_data = json.loads(response.text)
    generated_text = response_data[0]["generated_text"]
    print(generated_text)
    return generated_text


def text2speech(payload):
  API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
  headers = {"Authorization": f"Bearer {api_key}"}
  payload = {"inputs": payload}
  response = requests.post(API_URL, headers=headers, json=payload)
  with open('audio.flac','wb') as file:
    file.write(response.content)

#streamlit app
def main():
   st.set_page_config(page_title="Image to Speech", page_icon="📸", layout="centered", initial_sidebar_state="expanded")
   st.header("Image to audio story")
   uploaded_file = st.file_uploader("Choose an image...", type="jpg")

   if uploaded_file is not None:
       st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
       print(uploaded_file)
       bytes_data = uploaded_file.getvalue()
       with open(uploaded_file.name, "wb") as file:
           file.write(bytes_data)
       st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
       scenario = img2text(uploaded_file.name)
       story = text_to_story(scenario)
       text2speech(story)

       with st.expander("scenario"):
          st.write(scenario)
       with st.expander("story"):
          st.write(story)
       st.audio('audio.flac', format='audio/ogg', start_time=0) 

if __name__ == "__main__":
    main() 


