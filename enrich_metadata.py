import pandas as pd
from musicnn.tagger import top_tags
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

N_WORKERS = 24

from signal import signal, SIGSEGV



original_csv_path = "data/fma_metadata/csv_info_data.csv"
root_tags_path = "data/fma_metadata/tags/"
root_mp3_files = "data/fma_large/"
df = pd.read_csv(original_csv_path)

ids = df["id"].tolist()

def get_tags(file_path):
    
    warnings.filterwarnings(action= 'ignore')
    
    def handler(sigNum, frame):
        print("SEGMENTATION FAULT", sigNum)
    signal(SIGSEGV, handler)

    try:
        tags = top_tags(file_path, model='MTT_musicnn', topN=5, print_tags=False)
        f_name = file_path.split("/")[-1]
        f_name = f_name.replace(".mp3", ".txt")
        fw = open(root_tags_path + f_name, "w")
        fw.write(str(tags))
        fw.close()
    except Exception as e:
        print (e)
        tags = []
    return tags


file_paths = [root_mp3_files + "/" + f"{file_id:06d}"[0:3] + "/" + f"{file_id:06d}" + ".mp3" for file_id in ids]
#file_paths = file_paths[12900:]
r = Parallel(n_jobs=N_WORKERS)(delayed(get_tags)(fp) for fp in tqdm(file_paths))
df["tags"] = r

target_csv_path = "/data1/mlaquatra/TSOAI_hack/data/fma_metadata/csv_info_data_tags.csv"
df.to_csv(target_csv_path)

