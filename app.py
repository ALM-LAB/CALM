import streamlit as st
import numpy as np

from st_btn_select import st_btn_select
from streamlit_option_menu import option_menu

from cgi import test
import streamlit as st
import pandas as pd
from PIL import Image
import os
import glob

from transformers import CLIPVisionModel, AutoTokenizer, AutoModel
from transformers import ViTFeatureExtractor, ViTModel

import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from transformers import default_data_collator

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

from webcam import webcam

## Global Variables
MP3_ROOT_PATH = "/data2/akoudounas/fma/fma_large/"

IMAGE_SIZE = 224
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

TEXT_MODEL = 'bert-base-uncased'

## NavBar 
def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Text", "Audio", "Camera"],  # required
                icons=["chat-text", "mic", "camera"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Text", "Audio", "Camera"],  # required
            icons=["chat-text", "mic", "camera"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Text", "Audio", "Camera"],  # required
            icons=["chat-text", "mic", "camera"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#ffde59", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#5271ff"},
            },
        )
        return selected


## Draw Sidebar
def draw_sidebar(
    key,
    plot=False,
):

    st.write(
        """
        # Sidebar
        
        ```python
        Think.
        Search.
        Feel.
        ```
        """
    )

    st.slider("From 1 to 10, how cool is this app?", min_value=1, max_value=10, key=key)

    option = st_btn_select(('option1', 'option2', 'option3'), index=2)
    st.write(f'Selected option: {option}')

## Change Color
#def change_color(styles="")

## VisionDataset
class VisionDataset(Dataset):
    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    def __init__(self, image_paths: list):
        self.image_paths = image_paths

    def __getitem__(self, idx):
        return self.preprocess(Image.open(self.image_paths[idx]).convert('RGB'))

    def __len__(self):
        return len(self.image_paths)

## TextDataset
class TextDataset(Dataset):
    def __init__(self, text: list, tokenizer, max_len):
        self.len = len(text)
        self.tokens = tokenizer(text, padding='max_length',
                                max_length=max_len, truncation=True)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        return {'input_ids': token.ids, 'attention_mask': token.attention_mask}

    def __len__(self):
        return self.len

## CLIP Demo
class CLIPDemo:
    def __init__(self, vision_encoder, text_encoder, tokenizer,
                batch_size: int = 64, max_len: int = 64, device='cuda'):
        """ Initializes CLIPDemo
            it has the following functionalities:
                image_search: Search images based on text query
                zero_shot: Zero shot image classification
                analogy: Analogies with embedding space arithmetic.

            Args:
            vision_encoder: Fine-tuned vision encoder
            text_encoder: Fine-tuned text encoder
            tokenizer: Transformers tokenizer
            device (torch.device): Running device
            batch_size (int): Size of mini-batches used to embeddings
            max_length (int): Tokenizer max length

            Example:
            >>> demo = CLIPDemo(vision_encoder, text_encoder, tokenizer)
            >>> demo.compute_image_embeddings(test_df.image.to_list())
            >>> demo.image_search('یک مرد و یک زن')
            >>> demo.zero_shot('./workers.jpg')
            >>> demo.anology('./sunset.jpg', additional_text='دریا')
        """
        self.vision_encoder = vision_encoder.eval().to(device)
        self.text_encoder = text_encoder.eval().to(device)
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_embeddings_ = None
        self.image_embeddings_ = None
        

    def compute_image_embeddings(self, image_paths: list):
        self.image_paths = image_paths
        dataloader = DataLoader(VisionDataset(
            image_paths=image_paths), batch_size=self.batch_size, num_workers=8)
        embeddings = []
        with torch.no_grad():
            
            bar = st.progress(0)
            for i, images in tqdm(enumerate(dataloader), desc='computing image embeddings'):
                bar.progress(int(i/len(dataloader)*100))
                image_embedding = self.vision_encoder(
                    pixel_values=images.to(self.device)).pooler_output
                embeddings.append(image_embedding)
            bar.empty()
        self.image_embeddings_ =  torch.cat(embeddings)

    def compute_text_embeddings(self, text: list):
        self.text = text
        dataloader = DataLoader(TextDataset(text=text, tokenizer=self.tokenizer, max_len=self.max_len),
                                batch_size=self.batch_size, collate_fn=default_data_collator)
        embeddings = []
        with torch.no_grad():
            for tokens in tqdm(dataloader, desc='computing text embeddings'):
                image_embedding = self.text_encoder(input_ids=tokens["input_ids"].to(self.device),
                                                    attention_mask=tokens["attention_mask"].to(self.device)).pooler_output
                embeddings.append(image_embedding)
        self.text_embeddings_ = torch.cat(embeddings)

    def text_query_embedding(self, query: str = 'A happy song'):
        tokens = self.tokenizer(query, return_tensors='pt')
        with torch.no_grad():
            text_embedding = self.text_encoder(input_ids=tokens["input_ids"].to(self.device),
                                            attention_mask=tokens["attention_mask"].to(self.device)).pooler_output
        return text_embedding

    def most_similars(self, embeddings_1, embeddings_2):
        values, indices = torch.cosine_similarity(
            embeddings_1, embeddings_2).sort(descending=True)
        return values.cpu(), indices.cpu()


    def image_search(self, query: str, top_k=10):
        """ Search images based on text query
            Args:
                query (str): text query 
                image_paths (list[str]): a bunch of image paths
                top_k (int): number of relevant images 
        """
        query_embedding = self.text_query_embedding(query=query)
        _, indices = self.most_similars(self.image_embeddings_, query_embedding)

        matches = np.array(self.image_paths)[indices][:top_k]
        songs_path = []
        for match in matches:
            filename = os.path.split(match)[1]
            filename = int(filename.replace(".jpeg", ""))
            audio_path = MP3_ROOT_PATH + "/" + f"{filename:06d}"[0:3] + "/" + f"{filename:06d}"
            songs_path.append(audio_path)
        return songs_path

## Draw text page
def draw_text(
    key,
    plot=False,
):

    image = Image.open("data/logo.png")
    st.image(image, use_column_width="always")

    if 'model' not in st.session_state:
        #with st.spinner('We are orginizing your traks...'):
            text_encoder = AutoModel.from_pretrained("calm_spectrogram/best_text_model/", local_files_only=True)
            vision_encoder = CLIPVisionModel.from_pretrained("calm_spectrogram/best_vision_model/", local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
            model = CLIPDemo(vision_encoder=vision_encoder, text_encoder=text_encoder, tokenizer=tokenizer)
            model.compute_image_embeddings(glob.glob("spectrograms/*.jpeg"))
            st.session_state["model"] = model
            #st.session_state['model'] = CLIPDemo(vision_encoder=vision_encoder, text_encoder=text_encoder, tokenizer=tokenizer)
            #st.session_state.model.compute_image_embeddings(glob.glob("/data1/mlaquatra/TSOAI_hack/data/spectrograms/*.jpeg")[:100])
        #st.success('Done!')

    ""
    ""

    moods = ['-', 'angry', 'calm', 'happy', 'sad']
    genres = ['-', 'house', 'pop', 'rock', 'techno']
    artists = ['-', 'bad dad', 'lazy magnet', 'the astronauts', 'yan yalego']
    years = ['-', '80s', '90s', '2000s', '2010s']

    col1, col2 = st.columns(2)
    mood = col1.selectbox('Which mood do you feel right now?', moods, help="Select a mood here")
    genre = col2.selectbox('Which genre do you want to listen?', genres, help="Select a genre here")
    artist = col1.selectbox('Which artist do you like best?', artists, help="Select an artist here")
    year = col2.selectbox('Which period do you want to relive?', years, help="Select a period here")
    button_form = st.button('Search', key="button_form")

    st.text_input("Otherwise, describe the song you are looking for!", value="", key="sentence")
    button_sentence = st.button('Search', key="button_sentence")
        
    if (button_sentence and st.session_state.sentence != "") or (button_form and not (mood == "-" and artist == "-" and genre == "-" and year == "-")):
        if button_sentence:
            sentence = st.session_state.sentence    
        elif button_form:
            sentence = mood if mood != "-" else ""
            sentence = sentence + " " + genre if genre != "-" else sentence
            sentence = sentence + " " + artist if artist != "-" else sentence
            sentence = sentence + " " + year if year != "-" else sentence

        song_paths = st.session_state.model.image_search(sentence)
        for song in song_paths:
            st.audio(song + ".mp3", format="audio/mp3", start_time=0)

    if st.session_state.sentence == "Moreno La Quatra è un buffone":
        st.audio("data/005020.mp3", format="audio/mp3", start_time=0)

## Draw audio page
def draw_audio(
    key,
    plot=False,
):

    image = Image.open("data/logo.png")
    st.image(image, use_column_width="always")

    if 'model' not in st.session_state:
        #with st.spinner('We are orginizing your traks...'):
            text_encoder = AutoModel.from_pretrained("calm_spectrogram/best_text_model/", local_files_only=True)
            vision_encoder = CLIPVisionModel.from_pretrained("calm_spectrogram/best_vision_model/", local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
            model = CLIPDemo(vision_encoder=vision_encoder, text_encoder=text_encoder, tokenizer=tokenizer)
            model.compute_image_embeddings(glob.glob("spectrograms/*.jpeg")[:5000])
            st.session_state["model"] = model
            #st.session_state['model'] = CLIPDemo(vision_encoder=vision_encoder, text_encoder=text_encoder, tokenizer=tokenizer)
            #st.session_state.model.compute_image_embeddings(glob.glob("/data1/mlaquatra/TSOAI_hack/data/spectrograms/*.jpeg")[:100])
        #st.success('Done!')

    ""
    ""

    st.write("Please, describe the kind of song you are looking for!")
    stt_button = Button(label="Start Recording", margin=[5,5,5,200], width=200, default_size=10, width_policy='auto', button_type='primary')

    stt_button.js_on_event("button_click", CustomJS(code="""
        var recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = true;
    
        recognition.onresult = function (e) {
            var value = "";
            for (var i = e.resultIndex; i < e.results.length; ++i) {
                if (e.results[i].isFinal) {
                    value += e.results[i][0].transcript;
                }
            }
            if ( value != "") {
                document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
            }
        }
        recognition.start();
        """))

    result = streamlit_bokeh_events(
        stt_button,
        events="GET_TEXT",
        key="listen",
        refresh_on_update=False,
        override_height=75,
        debounce_time=0)
        
    if result:
        if "GET_TEXT" in result:
            sentence = result.get("GET_TEXT")
            st.write(sentence)

        song_paths = st.session_state.model.image_search(sentence)
        for song in song_paths:
            st.audio(song + ".mp3", format="audio/mp3", start_time=0)

## Draw camera page
def draw_camera(
    key,
    plot=False,
):

    image = Image.open("data/logo.png")
    st.image(image, use_column_width="always")

    if 'model' not in st.session_state:
        #with st.spinner('We are orginizing your traks...'):
            text_encoder = AutoModel.from_pretrained("calm_spectrogram/best_text_model/", local_files_only=True)
            vision_encoder = CLIPVisionModel.from_pretrained("calm_spectrogram/best_vision_model/", local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
            model = CLIPDemo(vision_encoder=vision_encoder, text_encoder=text_encoder, tokenizer=tokenizer)
            model.compute_image_embeddings(glob.glob("spectrograms/*.jpeg")[:500])
            st.session_state["model"] = model
            #st.session_state['model'] = CLIPDemo(vision_encoder=vision_encoder, text_encoder=text_encoder, tokenizer=tokenizer)
            #st.session_state.model.compute_image_embeddings(glob.glob("/data1/mlaquatra/TSOAI_hack/data/spectrograms/*.jpeg")[:100])
        #st.success('Done!')

    ""
    ""

    st.write("Please, show us how you are feeling today!")
    captured_image = webcam()
    if captured_image is None:
        st.write("Waiting for capture...")
    else:
        st.write("Got an image from the webcam:")
        
        st.image(captured_image)

        st.write(type(captured_image))
        st.write(captured_image)
        st.write(captured_image.size)
    

        vit_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        vit_model = ViTModel.from_pretrained("/data1/mlaquatra/TSOAI_hack/ViT_ER/best_checkpoint", local_files_only=True)
        inputs = vit_feature_extractor(images=captured_image, return_tensors="pt")
        outputs = vit_model(**inputs, output_hidden_states=True)
        st.write(outputs)


## Main 
selected = streamlit_menu(example=3)

if selected == "Text":
    # st.title(f"You have selected {selected}")
    draw_text("text", plot=True)
if selected == "Audio":
    # st.title(f"You have selected {selected}")
    draw_audio("audio", plot=True)
if selected == "Camera":
    # st.title(f"You have selected {selected}")
    draw_camera("camera", plot=True)

# with st.sidebar:
#     draw_sidebar("sidebar")
