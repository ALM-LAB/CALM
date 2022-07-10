import pandas as pd
from tqdm import tqdm

csv_file = pd.read_csv("../metadata/fma_metadata/raw_tracks.csv")
print(csv_file.head())
csv_file = csv_file.drop(columns=['album_title', 'album_url', 'artist_id',
    'artist_url', 'artist_website', 'license_image_file',
       'license_image_file_large', 'license_parent_id', 'license_title',
       'license_url', 'track_bit_rate', 'track_comments',
       'track_composer', 'track_copyright_c', 'track_copyright_p',
       'track_date_created', 'track_date_recorded', 'track_disc_number',
       'track_duration', 'track_explicit', 'track_explicit_notes',
       'track_favorites', 'track_file', 'track_image_file',
       'track_information', 'track_instrumental', 'track_interest',
       'track_language_code', 'track_listens', 'track_lyricist',
       'track_number', 'track_publisher', 'track_title', 'track_url'])


album_file = pd.read_csv("../metadata/fma_metadata/raw_albums.csv")
print(album_file.columns)

years = []
for id in tqdm(list(csv_file["album_id"])):
    row = album_file.loc[album_file["album_id"] == id]

    year = list(row["album_date_released"])[0] if len(list(row["album_date_released"])) > 0 else ""
    years.append(year)

csv_file["year"] = years
csv_file.to_csv("metadata.csv", index=False)