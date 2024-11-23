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