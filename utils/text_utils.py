import random
from .constants import *
import numpy as np

PROB_TRUE = 0.8

def get_standard_template(genre_list=None, 
                            mood_list=None, 
                            artist_list=None, 
                            year_list=None,
                            randomize = False):
    
    template_type = ""
    if artist_list is not None and artist_list != []:
        if randomize:
            take_it = np.random.choice([True, False], 1, p=[PROB_TRUE, 1-PROB_TRUE])[0]
            if take_it:
                template_type += "A"
        else:
            template_type += "A"

    if genre_list is not None and genre_list != []:
        if randomize:
            take_it = np.random.choice([True, False], 1, p=[PROB_TRUE, 1-PROB_TRUE])[0]
            if take_it:
                template_type += "G"
        else:
            template_type += "G"
       
    if mood_list is not None and mood_list != []:
        if randomize:
            take_it = np.random.choice([True, False], 1, p=[PROB_TRUE, 1-PROB_TRUE])[0]
            if take_it:
                template_type += "M"
        else:
            template_type += "M"

    if year_list is not None and year_list != []:
        if randomize:
            take_it = np.random.choice([True, False], 1, p=[PROB_TRUE, 1-PROB_TRUE])[0]
            if take_it:
                template_type += "Y"
        else:
            template_type += "Y"

    if len(template_type) == 0:
        if artist_list is not None and artist_list != []: template_type += "A"
        if genre_list is not None and genre_list != []: template_type += "G"
        if mood_list is not None and mood_list != []: template_type += "M"
        if year_list is not None and year_list != []: template_type += "Y"

    template = random.choice(STANDARD_TEMPLATES[template_type])

    if artist_list is not None and artist_list != []:
        selected_artist = random.choice(artist_list)
        if selected_artist[0].lower() in "aeiou" and " a [artist]" in " " + template.lower():
            template = template.replace(" [ARTIST]", "n " + selected_artist)
        else:
            template = template.replace("[ARTIST]", selected_artist)

    if genre_list is not None and genre_list != []:
        selected_genre = random.choice(genre_list)
        if selected_genre[0].lower() in "aeiou" and " a [genre]" in " " + template.lower():
            template = template.replace(" [GENRE]", "n " + selected_genre)
        else:
            template = template.replace("[GENRE]", selected_genre)
       
    if mood_list is not None and mood_list != []:
        selected_mood = random.choice(mood_list)
        if selected_mood[0].lower() in "aeiou" and " a [mood]" in " " + template.lower():
            template = template.replace(" [MOOD]", "n " + selected_mood)
        else:
            template = template.replace("[MOOD]", selected_mood)

    if year_list is not None and year_list != []:
        selected_year = random.choice(year_list)
        
        if selected_year >= 2000:
            decade = str(selected_year)[2]
            decade = "20" + str(decade) + "0s"
        elif 1910 <= selected_year < 2000:
            decade = str(selected_year)[2]
            decade += "0s"
        else:
            decade = str(selected_year)[1]
            decade = decade + "00"
            
        template = template.replace("[YEAR]", str(selected_year))

    return template.strip()
