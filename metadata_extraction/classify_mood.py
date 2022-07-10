import os
import tensorflow as tf
import librosa
import numpy as np
import shutil
import glob
import pandas as pd
from tqdm import tqdm
import warnings 
from joblib import Parallel, delayed

N_WORKERS = 24
root_mp3_files = "../data/fma_large/"

def get_logits(row, col):
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_resource_variables()
    
    tf.compat.v1.reset_default_graph()
    X = tf.compat.v1.placeholder(tf.float32, [None, row * col])
    X_img = tf.reshape(X, [-1, row, col, 1])

    W1 = tf.Variable(tf.random.normal([2, 4, 1, 32], stddev=0.01))
    L1 = tf.nn.conv2d(input=X_img, filters=W1, strides=[1, 1, 2, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool2d(input=L1, ksize=[1, 1, 3, 1],
                        strides=[1, 1, 3, 1], padding='SAME')

    W2 = tf.Variable(tf.random.normal([2, 4, 32, 64], stddev=0.01))
    L2 = tf.nn.conv2d(input=L1, filters=W2, strides=[1, 1, 2, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool2d(input=L2, ksize=[1, 1, 3, 1],
                        strides=[1, 1, 3, 1], padding='SAME')

    W3 = tf.Variable(tf.random.normal([2, 4, 64, 128], stddev=0.01))
    L3 = tf.nn.conv2d(input=L2, filters=W3, strides=[1, 1, 2, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool2d(input=L3, ksize=[1, 1, 3, 1],
                        strides=[1, 1, 3, 1], padding='SAME')
    L3_flat = tf.reshape(L3, [-1, 20 * 12 * 128])

    W4 = tf.compat.v1.get_variable("W4", use_resource = False, shape=[20 * 12 * 128, 4],
                            initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
    b = tf.Variable(tf.random.normal([4]))
    logits = tf.matmul(L3_flat, W4) + b
    return X, logits

def classify_mood (list_file_ids, model_path):

    def single_exec(fp):
        try:
            warnings.filterwarnings(action= 'ignore')
            y, sr = librosa.load(fp, duration=30)
            S = librosa.feature.melspectrogram(y=y, sr=12000, S=None, n_fft=512, hop_length=256, n_mels=96, fmax=8000)
            MF = librosa.feature.mfcc(S=librosa.power_to_db(S))
            row = MF.shape[0]
            col = MF.shape[1]
            MFCC_data = MF.flatten()
            MFCC_data = MFCC_data[np.newaxis, :]
            #MFCC_data = MFCC_data[:,:51640]
            
            X, logits = get_logits(row, col)

            with tf.compat.v1.Session() as sess:
                saver = tf.compat.v1.train.Saver()
                saver.restore(sess, model_path + "model.ckpt")
                result = sess.run(tf.argmax(input=logits, axis=1), feed_dict={X: MFCC_data[0:1]})

                if(result == 0):
                    mood = "happy"
                elif(result == 1):
                    mood = "angry"
                elif (result == 2):
                    mood = "sad"
                elif (result == 3):
                    mood = "calm"

            return mood
        except Exception as e:
            print (e)
            return None

    file_paths = [root_mp3_files + "/" + f"{file_id:06d}"[0:3] + "/" + f"{file_id:06d}" + ".mp3" for file_id in list_file_ids]
    r = Parallel(n_jobs=N_WORKERS)(delayed(single_exec)(fp) for fp in tqdm(file_paths))
    return r



original_csv_path = "../data/fma_metadata/csv_info_data.csv"
df = pd.read_csv(original_csv_path)
file_ids = df["id"].tolist()
#file_ids = file_ids[:100]
moods = classify_mood(file_ids, "Music_Mood_Classifier/model/mfcc/")
df["moods"] = moods
target_csv_path = "/data1/mlaquatra/TSOAI_hack/data/fma_metadata/csv_info_data_moods.csv"
df.to_csv(target_csv_path, index=False)
