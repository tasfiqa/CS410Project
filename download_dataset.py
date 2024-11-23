import kagglehub
import shutil
import os

def download_kaggle_dataset(handle):
    path = kagglehub.dataset_download(handle)
    handle = handle.replace("/", "_")
    local_path = f"data/{handle}"
    os.makedirs(path, exist_ok=True)
    shutil.move(path, local_path)

datasets = [
    "bahramjannesarr/goodreads-book-datasets-10m", 
    "jealousleopard/goodreadsbooks", 
    "arashnic/book-recommendation-dataset"
]

for dataset in datasets: 
    download_kaggle_dataset(dataset)
    print(f"Successfully downloaded {dataset}")
    