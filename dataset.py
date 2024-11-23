import kagglehub
import shutil
import os
import pandas as pd

def download_kaggle_dataset(handle):
    path = kagglehub.dataset_download(handle)
    handle = handle.replace("/", "_")
    local_path = f"data/{handle}"
    os.makedirs(path, exist_ok=True)
    shutil.move(path, local_path)

def download_datasets(): 
    datasets = [
    "arashnic/book-recommendation-dataset", 
    # "bahramjannesarr/goodreads-book-datasets-10m", 
    # "jealousleopard/goodreadsbooks", 
]

    for dataset in datasets: 
        download_kaggle_dataset(dataset)
        print(f"Successfully downloaded {dataset}")
    
def load_arashnic_dataset(
        df_path : str = f"data/arashnic_book-recommendation-dataset/Books.csv"
        ) -> pd.DataFrame: 
    return pd.read_csv(df_path)
