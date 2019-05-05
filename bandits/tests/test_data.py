import urllib.request as request
import urllib
import os
import sys
import pandas as pd
from os.path import dirname as d
from os.path import abspath, join
root_dir = d(d(d(abspath(__file__))))
sys.path.append(root_dir)

from bandits.data.data_preprocessor import download_data, one_hot_encoder

def test_download_data():
    file_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    file_path = os.path.join("datasets","mushroom.data")
    site = urllib.request.urlopen(file_url)
    download_data(file_url, file_name="mushroom.data")
    f = open(file_path, "rb")
    file_size = len(f.read())
    f.close()
    os.remove(file_path)
    try:
        os.removedirs("datasets")
    except:
        pass
    assert site.length == file_size

def test_one_hot_encoder():
    test_df = pd.DataFrame()
    test_df['col1'] = list(range(3))*10
    encoded_df = pd.DataFrame()
    encoded_df['col1 0'] = [1,0,0]*10
    encoded_df['col1 1'] = [0,1,0]*10
    encoded_df['col1 2'] = [0,0,1]*10
    module_encoded_df = one_hot_encoder(test_df, ['col1'])
    for i in range(len(module_encoded_df.columns)):
        module_encoded_col = module_encoded_df.columns[i]
        encoded_col = encoded_df.columns[i]
        assert module_encoded_df[module_encoded_col].astype(type(encoded_df[encoded_col][0])).equals(encoded_df[encoded_col])