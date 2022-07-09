import pandas as pd

root_tags_path = "data/fma_metadata/tags/"
f1 = "data/fma_metadata/csv_info_data_moods.csv"
f = "data/fma_metadata/csv_info_data_final.csv"

df_1 = pd.read_csv(f1)
list_ids = df_1["id"].tolist()
list_tags = []

for file_id in list_ids:
    try:
        fr = open(root_tags_path + f"{file_id:06d}" + ".txt", "r")
        text = fr.read()
        fr.close()
        tags = eval(text)
        list_tags.append(tags)
    except Exception as e:
        list_tags.append([])

df_1["tags"] = list_tags
df_1.drop(columns=["Unnamed: 0"], inplace=True)
df_1.to_csv(f, index=False)
