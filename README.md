# CS410Project
UIUC CS 410 Project for Fall 2024

## Overview

This project is a book recommendation system that leverages machine learning models to provide personalized book suggestions. The system uses a dataset from Kaggle and processes it to generate embeddings for book titles, which are then used to compute similarities and make recommendations.

## Table of Contents

- [CS410Project](#cs410project)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Model](#model)
    - [Model and Embeddings](#model-and-embeddings)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)
  - [Contact](#contact)
    - [Code References](#code-references)

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/CS410Project.git
   cd CS410Project
   ```

2. **Install the required packages:**

   Ensure you have Python 3.11.4 installed. Then, install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes essential libraries such as `pandas`, `transformers`, `sentence-transformers`, and `tqdm`.

3. **Download the dataset:**

   The dataset is downloaded from Kaggle using the `download_dataset.py` script. Ensure you have Kaggle API credentials set up.

   ```bash
   python download_dataset.py
   ```

   This script downloads datasets from Kaggle and moves them to the `data/` directory.

## Dataset

The project uses the "Book Recommendation Dataset" from Kaggle. The dataset is stored in the `data/arashnic_book-recommendation-dataset` directory. The dataset includes information about books such as ISBN, title, author, year of publication, and publisher.

## Model

The project uses the `multi-qa-mpnet-base-cos-v1` model from the `sentence-transformers` library to generate embeddings for book titles. The model is either loaded from a local directory or downloaded and saved if not already present.

### Model and Embeddings

Due to size constraints, the model and embeddings are stored on Google Drive. You can download them using the following links:

- [Model Download Link]([https://drive.google.com/drive/folders/1S2yuk7m_2cgG6pjHyNwibz-BRyxlrL-e?usp=drive_link])
- [Embeddings Download Link]([https://drive.google.com/file/d/1472tKVjS0GwDQcjWBFOJhXn6WF8zDUK_/view?usp=drive_link])

After downloading, place the model in the `models/` directory and the embeddings in the `data/embeddings/` directory.

## Usage

To generate embeddings and use the recommendation system, run the following script:

```bash
python embeddings.py
```


This script performs the following tasks:

- Loads the book titles from the dataset.
- Converts titles to lowercase.
- Generates embeddings using the loaded model.
- Saves the embeddings to the specified directory.

For more detailed exploration and analysis, you can use the `eda.ipynb` Jupyter notebook, which includes data exploration and model usage examples.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset is sourced from Kaggle.
- The project uses the `sentence-transformers` library for generating embeddings.

## Contact


### Code References

- The dataset download script is referenced from:

```
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


- The model loading and embedding generation logic is referenced from:

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
