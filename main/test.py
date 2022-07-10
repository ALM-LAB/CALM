from utils.text_utils import *
from utils.audio_utils import *
from utils.utils import *
import os, glob
from tqdm import tqdm
from os.path import exists
import random

metadata_dict = load_from_file("data/fma_metadata/tracks_metadata_dict.pkl")
#genres_dict = load_from_file("data/fma_metadata/tracks_metadata_dict.pkl")

root_spectrograms = "data/spectrograms/"

existing_files = 0

keys = list(metadata_dict.keys())
random.shuffle(keys)

shuffled_dict = dict()
for key in keys:
    shuffled_dict.update({key: metadata_dict[key]})

metadata_dict = shuffled_dict

for k, v in tqdm(metadata_dict.items()):
    if exists(root_spectrograms + v["prefix_filename"] + ".jpeg"):
        existing_files += 1
        #print(root_spectrograms + v["prefix_filename"] + ".jpeg")
        template_type = ""

        # artist
        if v["artist_name"] is not None:
            artist_list = [v["artist_name"]]

        # genres
        if v["genres"] is not None:
            genres_list = []
            for d_genre in v["genres"]:
                genres_list.append(d_genre["genre_title"])

        # mood : TODO

        # year
        if v["year"] is not None:
            year_list = [v["year"]]

        # print (f"Artist: {artist_list}")
        # print (f"Genres: {genres_list}")
        # print (f"Year  : {year_list}")
        song_description = get_standard_template(genre_list=genres_list, 
                                                year_list=year_list, 
                                                artist_list=artist_list, 
                                                mood_list=[])
        print (song_description)

print(f"Total files: {existing_files}")

# print (get_standard_template(genre_list=["electro-house"], year_list=[2021]))
# print (get_standard_template(mood_list=["energetic"], year_list=[2021], artist_list=["Avicii", "David Guetta"]))
# print (get_standard_template(genre_list=["electro"], year_list=[2005], artist_list=["Martin Solveig"], mood_list=["funny"]))

