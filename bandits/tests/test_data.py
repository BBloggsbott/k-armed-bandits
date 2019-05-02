import urllib.request as request
import urllib
import os
import sys
from os.path import dirname as d
from os.path import abspath, join
root_dir = d(d(d(abspath(__file__))))
sys.path.append(root_dir)

print(sys.path)

from bandits.data.data_preprocessor import download_data

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