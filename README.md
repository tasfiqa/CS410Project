# CS410Project
Book Recommendation Project for UIUC CS 410 Project for Fall 2024

## Overview

This project is a book recommendation system that leverages machine learning models to provide personalized book suggestions. The system fetches a dataset from Kaggle, generates embeddings on the book titles and descriptions, then uses those embeddings to compute similarity scores against user queries and recommend the highest-scoring matches.

Tutorial for our software can be found [here](https://drive.google.com/file/d/1FL8jbLJN8gf84ziGvPR03j9v73bF-RnF/view?usp=sharing)

## Table of Contents

- [CS410Project](#cs410project)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Model](#model)
  - [Usage](#usage)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)
    - [Code References](#code-references)

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/CS410Project.git
   cd CS410Project
   ```

2. **Install the required packages:**

   Ensure you have Python 3.10.15 installed. Then, install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes essential libraries such as `pandas`, `transformers`, `sentence-transformers`, and `tqdm`.

3. **Download the dataset:**

   The python file script automatically downloads datasets from Kaggle and moves them to the `data/` directory.

## Dataset

This project utilises a Science Fiction Books dataset sourced from Kaggle. This dataset is stored in the `data/tanguypledel_science-fiction-books-subgenres` directory. The dataset is divided into twelve subcategories, each filtering the science fiction books by a differen subgenre. Each smaller dataset contains a variety of fields about the books within, including the title, author, edition, score, votes, reviews, descriptions, publication date, and a detailed list of genres.

Dataset can be found [here](https://www.kaggle.com/datasets/tanguypledel/science-fiction-books-subgenres)

## Model

The project uses the `all-MiniLM-L6-v2` model from the `sentence-transformers` library to generate embeddings based on book titles and descriptions. The model will be loaded from a local cache if available, and if not, it will be downloaded through the sentence_transformers library.

## Usage

To download dataset, generate embeddings, and use the recommendation system, run the following script:

```bash
python book_recommendation.py
```


This script performs the following tasks:

- Fetch and merge all subsets from the Science Fiction Books dataset.
- Load the book titles and descriptions from the dataset.
- Generate embeddings using the all-MiniLM-L6-v2 model.
- Save the embeddings to the specified directory.
- Ask the user to describe their desired book.
- Generate a vector embedding for the user's query.
- Search against the pre-computed embeddings for matches.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset is sourced from Kaggle.
- The project uses the `sentence-transformers` library for generating embeddings.

### Code References

- The dataset download script is referenced from:

```python
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
```


- The model loading and embedding generation logic is referenced from:

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os

def load_model(model : str ="multi-qa-mpnet-base-cos-v1") -> SentenceTransformer:
    model_path = f'models/{model}'
    if os.path.exists(model_path):
        return SentenceTransformer(model_path)

    # Save model if not saved
    os.makedirs(model_path)
    model = SentenceTransformer(model)
    model.save(model_path)
    return model

def get_embeddings(model : SentenceTransformer, values : np.ndarray) -> np.ndarray:
    return model.encode_multi_process(values, show_progress_bar=True)

if __name__ == "__main__":
    dataset = "arashnic_book-recommendation-dataset/"
    df_path = f"data/{dataset}/Books.csv"
    book_titles = pd.read_csv(df_path)['Book-Title']
    # Lowercase 
    book_titles = (
        book_titles
        .str.lower()
        .values
        )
    model = load_model()
    embeddings = get_embeddings(model, book_titles)
    embeddings_write_path = f"data/embeddings/{dataset}"
    if not os.path.exists(embeddings_write_path):
        os.makedirs(embeddings_write_path)
        np.save(f"{embeddings_write_path}/embeddings.npy")
```
