from sre_parse import Tokenizer
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from utils.text_utils import *
from utils.utils import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from transformers import CLIPModel, CLIPConfig, CLIPVisionModel
from transformers import CLIPFeatureExtractor
from transformers import AutoModel, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import default_data_collator

from torch.cuda.amp import autocast
from torch import nn
import torch

IMAGE_SIZE = 224
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

TEST_SIZE = 0.2
SEED = 42
PROB_TRUE = 0.2

ID_COLUMN_NAME = "id"
DATA_FILE = "data/fma_metadata/csv_info_data.csv"
SPECTROGRAM_PATH = "data/spectrograms/"
#SPECTROGRAM_PATH = "data/chroma/"


IMAGE_MODEL = 'openai/clip-vit-base-patch32'
TEXT_MODEL = 'bert-base-uncased'
MAX_LEN = 64
BATCH_SIZE = 64
N_EPOCHS = 2
CHECKPOINT_PATH = "calm_spectrogram"

def clip_wraper_creator():
    """create a dummy CLIPModel to wrap text and vision encoders in order to use CLIPTrainer"""
    config = {'num_hidden_layers': 0,
              'max_position_embeddings': 0,
              'vocab_size': 0,
              'hidden_size': 1,
              'patch_size': 1,
              }
    DUMMY_CONFIG = CLIPConfig(text_config_dict=config,
                              vision_config_dict=config)
    clip = CLIPModel(config=DUMMY_CONFIG)
    # convert projectors to Identity
    clip.text_projection = nn.Identity()
    clip.visual_projection = nn.Identity()
    return clip


class CLIPDataset(Dataset):
    def __init__(self, file_ids: list, # ["000003", "000231"]
                        root_path_images: str, 
                        dataframe_text: pd.DataFrame,
                        tokenizer: AutoTokenizer):
        self.file_ids = file_ids
        self.dataframe_text = dataframe_text
        self.root_path_images = root_path_images
        
        #self.tokens = tokenizer(text, padding='max_length', max_length=MAX_LEN, truncation=True)
        self.tokenizer = tokenizer

        self.augment = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

    def __getitem__(self, idx):

        file_id = int(self.file_ids[idx])
        filename_image = f"{file_id:06d}" + ".jpeg"
        # get image
        image_path = os.path.join(self.root_path_images, filename_image)
        pixel_values = self.augment(Image.open(image_path).convert('RGB'))
        
        # get text
        row = self.dataframe_text.loc[self.dataframe_text[ID_COLUMN_NAME] == file_id]

        genre = eval(list(row["genres"])[0])
        year = [list(row["year"])[0]]
        artist = [list(row["artist"])[0]]
        mood = [list(row["mood"])[0]]
        tags = eval(list(row["tags"])[0])

        use_tags = np.random.choice([True, False], 1, p=[PROB_TRUE, 1-PROB_TRUE])[0]
        if len(tags) > 0 and use_tags:
            text_values = " ".join(tags)
            text_values = text_values.strip()
        else:
            text_values = get_standard_template(genre_list=genre, 
                                                mood_list=mood, 
                                                artist_list=artist, 
                                                year_list=year, 
                                                randomize=True)

        inputs = self.tokenizer(text_values, padding='max_length', max_length=MAX_LEN, truncation=True, return_tensors="pt")
        

        #token = self.tokens[idx]
        return {
                'input_ids': inputs["input_ids"][0],
                'attention_mask': inputs["attention_mask"][0],
                'pixel_values': pixel_values
                }

    def __len__(self):
        return len(self.file_ids)


class CLIPTrainer(Trainer):

    def set_use_amp(self, use_amp=True):
        self.use_amp = True


    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs, return_loss=True)
        return outputs["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    loss = self.compute_loss(model, inputs)
            else:
                loss = self.compute_loss(model, inputs)
        return (loss, None, None)
    



if __name__ == '__main__':
    df = pd.read_csv(DATA_FILE)

    df = df.fillna("None")
    
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED)
    eval_df, test_df = train_test_split(test_df, test_size=0.5, random_state=SEED)

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)

    train_ds = CLIPDataset(file_ids=train_df[ID_COLUMN_NAME].tolist(),
                            root_path_images=SPECTROGRAM_PATH,
                            dataframe_text=train_df,
                            tokenizer=tokenizer)
    eval_ds = CLIPDataset(file_ids=eval_df[ID_COLUMN_NAME].tolist(),
                            root_path_images=SPECTROGRAM_PATH,
                            dataframe_text=eval_df,
                            tokenizer=tokenizer)
    test_ds = CLIPDataset(file_ids=test_df[ID_COLUMN_NAME].tolist(),
                            root_path_images=SPECTROGRAM_PATH,
                            dataframe_text=test_df,
                            tokenizer=tokenizer)

    '''
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          collate_fn=default_data_collator)

    for item in train_dl:
        print (item)
        print(item['input_ids'].shape)
        print(item['pixel_values'].shape)
        print(item['input_ids'])
        text = tokenizer.batch_decode(item['input_ids'], skip_special_tokens=True)
        print('Text: ', text)
        break
    '''

    vision_encoder = CLIPVisionModel.from_pretrained(IMAGE_MODEL)
    text_encoder = AutoModel.from_pretrained(TEXT_MODEL)

    clip = clip_wraper_creator()
    clip.text_model = text_encoder
    clip.vision_model = vision_encoder

    vision_preprocessor = CLIPFeatureExtractor.from_pretrained(IMAGE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)

    '''
    out = clip(input_ids=item['input_ids'],
               attention_mask=item['attention_mask'],
               pixel_values=item['pixel_values'],
               return_loss=True)

    print('text and image embeddings: ',
          out.text_embeds.shape, out.image_embeds.shape)
    print('loss: ', out.loss)
    del out, item
    '''

    args = TrainingArguments(
        CHECKPOINT_PATH,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        eval_steps=100,
        logging_steps=10,
        learning_rate=3e-5,
        weight_decay=0.003,
        warmup_steps=100,
        fp16=True,
        prediction_loss_only=True,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=N_EPOCHS,
        load_best_model_at_end=True,
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    clear_gpu()
    args.dataloader_num_workers = 4 #optimal_workers()
    trainer = CLIPTrainer(clip, args,
                          train_dataset=train_ds,
                          eval_dataset=test_ds)
    
    trainer.set_use_amp()

    trainer.train()
    trainer.save_model(CHECKPOINT_PATH + "/best_checkpoint/")

    trainer.model.vision_model.save_pretrained(CHECKPOINT_PATH + "/best_vision_model/")
    trainer.model.text_model.save_pretrained(CHECKPOINT_PATH + "/best_text_model/")

   
