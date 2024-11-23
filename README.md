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

- [Model Download Link](https://drive.google.com/your-model-link)
- [Embeddings Download Link](https://drive.google.com/your-embeddings-link)

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

For any questions or issues, please contact [your-email@example.com].


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


- The data exploration and analysis are demonstrated in:

```
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Arashnic"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\tasfi\\AppData\\Local\\Temp\\ipykernel_20984\\3961890407.py:3: DtypeWarning: Columns (3) have mixed types. Specify dtype option on import or set low_memory=False.\n",
      "  df = pd.read_csv(\"data/arashnic_book-recommendation-dataset/Books.csv\")\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Index(['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher',\n",
       "       'Image-URL-S', 'Image-URL-M', 'Image-URL-L'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd \n",
    "\n",
    "df = pd.read_csv(\"data/arashnic_book-recommendation-dataset/Books.csv\")\n",
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Lowercase column names \n",
    "df.columns = [col.lower() for col in df.columns]\n",
    "\n",
    "# Lowecase book titles\n",
    "df['book-titles'] = df['book-title'].str.lower()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a20af52ab0b64ad3ad394f3a80f65d4f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\tasfi\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\huggingface_hub\\file_download.py:139: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\\Users\\tasfi\\.cache\\huggingface\\hub\\models--sentence-transformers--multi-qa-mpnet-base-cos-v1. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.\n",
      "To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development\n",
      "  warnings.warn(message)\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9bfdc23807144d79b45dc74264c4d11c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "config_sentence_transformers.json:   0%|          | 0.00/116 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5c0c46ce3b28451d9de4144370ec769d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "README.md:   0%|          | 0.00/9.25k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5a5c31a1c6bc4109b5b03cad949aa98a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "sentence_bert_config.json:   0%|          | 0.00/53.0 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "15f7fd5aa4934f12827280bbc8e64470",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "config.json:   0%|          | 0.00/571 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e633bc2515d741d98dc1136f355fed89",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "model.safetensors:   0%|          | 0.00/438M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8428437ef0e943ac92752ed917bbaaa9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "tokenizer_config.json:   0%|          | 0.00/363 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8e3c981c71984f10bea1cce2901a86cb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "215fcc1c23a54708b9b5deddfe59e590",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "18179f5568314a4f899b8f50e9c835b1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "special_tokens_map.json:   0%|          | 0.00/239 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1416dc0c549d477c8f407f7cbdbda919",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "1_Pooling/config.json:   0%|          | 0.00/190 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sentence_transformers import SentenceTransformer\n",
    "\n",
    "model = SentenceTransformer(\"multi-qa-mpnet-base-cos-v1\")\n",
    "\n",
    "query_embedding = model.encode(\"How big is London\")\n",
    "passage_embeddings = model.encode([\n",
    "    \"London is known for its financial district\",\n",
    "    \"London has 9,787,426 inhabitants at the 2011 census\",\n",
    "    \"The United Kingdom is the fourth largest exporter of goods in the world\",\n",
    "])\n",
    "\n",
    "similarity = model.similarity(query_embedding, passage_embeddings)\n",
    "# => tensor([[0.4659, 0.6142, 0.2697]])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save(\"models/multi-qa-mpnet-base-cos-v1\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "model_path = \"models/multi-qa-mpnet-base-cos-v1\"\n",
    "model = SentenceTransformer(model_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['embedding'] = df['book-title'].apply(model.encode)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(5, 768)"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_vals = (df['book-title'].iloc[:5].values)\n",
    "model.encode(test_vals).shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>isbn</th>\n",
       "      <th>book-title</th>\n",
       "      <th>book-author</th>\n",
       "      <th>year-of-publication</th>\n",
       "      <th>publisher</th>\n",
       "      <th>image-url-s</th>\n",
       "      <th>image-url-m</th>\n",
       "      <th>image-url-l</th>\n",
       "      <th>book-titles</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0195153448</td>\n",
       "      <td>Classical Mythology</td>\n",
       "      <td>Mark P. O. Morford</td>\n",
       "      <td>2002</td>\n",
       "      <td>Oxford University Press</td>\n",
       "      <td>http://images.amazon.com/images/P/0195153448.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0195153448.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0195153448.0...</td>\n",
       "      <td>classical mythology</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0002005018</td>\n",
       "      <td>Clara Callan</td>\n",
       "      <td>Richard Bruce Wright</td>\n",
       "      <td>2001</td>\n",
       "      <td>HarperFlamingo Canada</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0002005018.0...</td>\n",
       "      <td>clara callan</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0060973129</td>\n",
       "      <td>Decision in Normandy</td>\n",
       "      <td>Carlo D'Este</td>\n",
       "      <td>1991</td>\n",
       "      <td>HarperPerennial</td>\n",
       "      <td>http://images.amazon.com/images/P/0060973129.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0060973129.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0060973129.0...</td>\n",
       "      <td>decision in normandy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0374157065</td>\n",
       "      <td>Flu: The Story of the Great Influenza Pandemic...</td>\n",
       "      <td>Gina Bari Kolata</td>\n",
       "      <td>1999</td>\n",
       "      <td>Farrar Straus Giroux</td>\n",
       "      <td>http://images.amazon.com/images/P/0374157065.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0374157065.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0374157065.0...</td>\n",
       "      <td>flu: the story of the great influenza pandemic...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0393045218</td>\n",
       "      <td>The Mummies of Urumchi</td>\n",
       "      <td>E. J. W. Barber</td>\n",
       "      <td>1999</td>\n",
       "      <td>W. W. Norton &amp;amp; Company</td>\n",
       "      <td>http://images.amazon.com/images/P/0393045218.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0393045218.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0393045218.0...</td>\n",
       "      <td>the mummies of urumchi</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>271355</th>\n",
       "      <td>0440400988</td>\n",
       "      <td>There's a Bat in Bunk Five</td>\n",
       "      <td>Paula Danziger</td>\n",
       "      <td>1988</td>\n",
       "      <td>Random House Childrens Pub (Mm)</td>\n",
       "      <td>http://images.amazon.com/images/P/0440400988.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0440400988.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0440400988.0...</td>\n",
       "      <td>there's a bat in bunk five</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>271356</th>\n",
       "      <td>0525447644</td>\n",
       "      <td>From One to One Hundred</td>\n",
       "      <td>Teri Sloat</td>\n",
       "      <td>1991</td>\n",
       "      <td>Dutton Books</td>\n",
       "      <td>http://images.amazon.com/images/P/0525447644.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0525447644.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0525447644.0...</td>\n",
       "      <td>from one to one hundred</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>271357</th>\n",
       "      <td>006008667X</td>\n",
       "      <td>Lily Dale : The True Story of the Town that Ta...</td>\n",
       "      <td>Christine Wicker</td>\n",
       "      <td>2004</td>\n",
       "      <td>HarperSanFrancisco</td>\n",
       "      <td>http://images.amazon.com/images/P/006008667X.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/006008667X.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/006008667X.0...</td>\n",
       "      <td>lily dale : the true story of the town that ta...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>271358</th>\n",
       "      <td>0192126040</td>\n",
       "      <td>Republic (World's Classics)</td>\n",
       "      <td>Plato</td>\n",
       "      <td>1996</td>\n",
       "      <td>Oxford University Press</td>\n",
       "      <td>http://images.amazon.com/images/P/0192126040.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0192126040.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0192126040.0...</td>\n",
       "      <td>republic (world's classics)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>271359</th>\n",
       "      <td>0767409752</td>\n",
       "      <td>A Guided Tour of Rene Descartes' Meditations o...</td>\n",
       "      <td>Christopher  Biffle</td>\n",
       "      <td>2000</td>\n",
       "      <td>McGraw-Hill Humanities/Social Sciences/Languages</td>\n",
       "      <td>http://images.amazon.com/images/P/0767409752.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0767409752.0...</td>\n",
       "      <td>http://images.amazon.com/images/P/0767409752.0...</td>\n",
       "      <td>a guided tour of rene descartes' meditations o...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>271360 rows × 9 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "              isbn                                         book-title  \\\n",
       "0       0195153448                                Classical Mythology   \n",
       "1       0002005018                                       Clara Callan   \n",
       "2       0060973129                               Decision in Normandy   \n",
       "3       0374157065  Flu: The Story of the Great Influenza Pandemic...   \n",
       "4       0393045218                             The Mummies of Urumchi   \n",
       "...            ...                                                ...   \n",
       "271355  0440400988                         There's a Bat in Bunk Five   \n",
       "271356  0525447644                            From One to One Hundred   \n",
       "271357  006008667X  Lily Dale : The True Story of the Town that Ta...   \n",
       "271358  0192126040                        Republic (World's Classics)   \n",
       "271359  0767409752  A Guided Tour of Rene Descartes' Meditations o...   \n",
       "\n",
       "                 book-author year-of-publication  \\\n",
       "0         Mark P. O. Morford                2002   \n",
       "1       Richard Bruce Wright                2001   \n",
       "2               Carlo D'Este                1991   \n",
       "3           Gina Bari Kolata                1999   \n",
       "4            E. J. W. Barber                1999   \n",
       "...                      ...                 ...   \n",
       "271355        Paula Danziger                1988   \n",
       "271356            Teri Sloat                1991   \n",
       "271357      Christine Wicker                2004   \n",
       "271358                 Plato                1996   \n",
       "271359   Christopher  Biffle                2000   \n",
       "\n",
       "                                               publisher  \\\n",
       "0                                Oxford University Press   \n",
       "1                                  HarperFlamingo Canada   \n",
       "2                                        HarperPerennial   \n",
       "3                                   Farrar Straus Giroux   \n",
       "4                             W. W. Norton &amp; Company   \n",
       "...                                                  ...   \n",
       "271355                   Random House Childrens Pub (Mm)   \n",
       "271356                                      Dutton Books   \n",
       "271357                                HarperSanFrancisco   \n",
       "271358                           Oxford University Press   \n",
       "271359  McGraw-Hill Humanities/Social Sciences/Languages   \n",
       "\n",
       "                                              image-url-s  \\\n",
       "0       http://images.amazon.com/images/P/0195153448.0...   \n",
       "1       http://images.amazon.com/images/P/0002005018.0...   \n",
       "2       http://images.amazon.com/images/P/0060973129.0...   \n",
       "3       http://images.amazon.com/images/P/0374157065.0...   \n",
       "4       http://images.amazon.com/images/P/0393045218.0...   \n",
       "...                                                   ...   \n",
       "271355  http://images.amazon.com/images/P/0440400988.0...   \n",
       "271356  http://images.amazon.com/images/P/0525447644.0...   \n",
       "271357  http://images.amazon.com/images/P/006008667X.0...   \n",
       "271358  http://images.amazon.com/images/P/0192126040.0...   \n",
       "271359  http://images.amazon.com/images/P/0767409752.0...   \n",
       "\n",
       "                                              image-url-m  \\\n",
       "0       http://images.amazon.com/images/P/0195153448.0...   \n",
       "1       http://images.amazon.com/images/P/0002005018.0...   \n",
       "2       http://images.amazon.com/images/P/0060973129.0...   \n",
       "3       http://images.amazon.com/images/P/0374157065.0...   \n",
       "4       http://images.amazon.com/images/P/0393045218.0...   \n",
       "...                                                   ...   \n",
       "271355  http://images.amazon.com/images/P/0440400988.0...   \n",
       "271356  http://images.amazon.com/images/P/0525447644.0...   \n",
       "271357  http://images.amazon.com/images/P/006008667X.0...   \n",
       "271358  http://images.amazon.com/images/P/0192126040.0...   \n",
       "271359  http://images.amazon.com/images/P/0767409752.0...   \n",
       "\n",
       "                                              image-url-l  \\\n",
       "0       http://images.amazon.com/images/P/0195153448.0...   \n",
       "1       http://images.amazon.com/images/P/0002005018.0...   \n",
       "2       http://images.amazon.com/images/P/0060973129.0...   \n",
       "3       http://images.amazon.com/images/P/0374157065.0...   \n",
       "4       http://images.amazon.com/images/P/0393045218.0...   \n",
       "...                                                   ...   \n",
       "271355  http://images.amazon.com/images/P/0440400988.0...   \n",
       "271356  http://images.amazon.com/images/P/0525447644.0...   \n",
       "271357  http://images.amazon.com/images/P/006008667X.0...   \n",
       "271358  http://images.amazon.com/images/P/0192126040.0...   \n",
       "271359  http://images.amazon.com/images/P/0767409752.0...   \n",
       "\n",
       "                                              book-titles  \n",
       "0                                     classical mythology  \n",
       "1                                            clara callan  \n",
       "2                                    decision in normandy  \n",
       "3       flu: the story of the great influenza pandemic...  \n",
       "4                                  the mummies of urumchi  \n",
       "...                                                   ...  \n",
       "271355                         there's a bat in bunk five  \n",
       "271356                            from one to one hundred  \n",
       "271357  lily dale : the true story of the town that ta...  \n",
       "271358                        republic (world's classics)  \n",
       "271359  a guided tour of rene descartes' meditations o...  \n",
       "\n",
       "[271360 rows x 9 columns]"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import torch\n",
    "torch.cuda.is_available()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "torch.cuda.device_count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "torch.cuda.get_arch_list()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import sys \n",
    "def in_venv():\n",
    "    return sys.prefix != sys.base_prefix\n",
    "\n",
    "in_venv()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

```