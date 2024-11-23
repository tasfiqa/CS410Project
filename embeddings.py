from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os
import torch

def load_model(model : str ="multi-qa-mpnet-base-cos-v1") -> SentenceTransformer: 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = f'models/{model}'
    if os.path.exists(model_path): 
        model = SentenceTransformer(model_path, device=device)
        print(f"Successfully loaded model onto {device}")
        return model
    # Save model if not saved
    os.makedirs(model_path)
    model = SentenceTransformer(model, device=device)
    print(f"Successfully loaded model onto {device}")
    model.save(model_path)
    print(f"Successfully saved model to {model_path}")
    return model 

def load_embeddings(
        embeddings_path : str ="data/arashnic_book-recommendation-dataset/embeddings.npy"
        ) -> np.ndarray: 
    return np.load(embeddings_path)
    
def get_embeddings(model : SentenceTransformer, values : np.ndarray) -> np.ndarray: 
    return model.encode(values, show_progress_bar=True)

def write_embeddings() -> None: 
    dataset = "arashnic_book-recommendation-dataset/"
    df_path = f"data/{dataset}/Books.csv"
    book_titles = pd.read_csv(df_path)['Book-Title']
    # Lowercase 
    book_titles = (
        book_titles
        .str.lower()
        .values
        )
    print(f"cuda : {torch.cuda.is_available()}")
    model = load_model()
    embeddings = get_embeddings(model, book_titles)
    with open(f"data/{dataset}/embeddings.npy", mode='wb') as f: 
        np.save(
            file=f, 
            arr=embeddings,
            )

if __name__ == "__main__": 
    pass 