"""Functions to download, preprocess data and create bandit problems"""

import urllib.request
import os
import validators

def download_data(file_url, download_dir = "datasets", file_name = None):
    """
    This function downloads a file over HTTP
    Args:
        file_url: The url of the file to download
        download_dir: The sirectory to save the downloaded files to
        file_name: Name of the downloaded file. Uses the file name from the url by default      
    """
    if not validators.url(file_url):
        raise RuntimeError("Invalid URL {}".format(file_url))
    
    if file_name == None:
        file_name = file_url.split('/')[-1]
    
    if not os.path.isdir(download_dir):
        os.makedirs(download_dir)
    
    urllib.request.urlretrieve(file_url, os.path.join(download_dir, file_name))