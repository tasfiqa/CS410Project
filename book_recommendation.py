import kagglehub
import os
import pandas as pd
import shutil
import torch
from sentence_transformers import SentenceTransformer, util

datasets = [
    "tanguypledel/science-fiction-books-subgenres"
]

model = SentenceTransformer('all-MiniLM-L6-v2')

def download_kaggle_dataset(handle):
    path = kagglehub.dataset_download(handle)
    handle = handle.replace("/", "_")
    local_path = f"data/{handle}"
    os.makedirs(path, exist_ok=True)
    shutil.move(path, local_path)

def download():
    for dataset in datasets:
        download_kaggle_dataset(dataset)
        print(f"Successfully downloaded {dataset}")

def combine_csv_files(input_folder, output_file):
    # Join all sub-sets from the dataset, then remove duplicate entries
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    dataframes = []
    for csv_file in csv_files:
        file_path = os.path.join(input_folder, csv_file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df['Book_Title'] = combined_df['Book_Title'].str.strip()
    combined_df = combined_df.drop_duplicates(subset=['Book_Title'])
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")


def generate_embeddings(combined_csv_path):
    # Generate embeddings using the book descriptions
    # Embeddings are saved to a file for later usage
    df = pd.read_csv(combined_csv_path)
    if 'Book_Description' not in df.columns or 'Book_Title' not in df.columns:
        raise ValueError("The dataset must contain 'description' and 'title' columns.")
    descriptions = df['Book_Description'].astype(str).tolist()
    book_embeddings = model.encode(descriptions, convert_to_tensor=True)
    embeddings_path = 'book_embeddings.pt'
    torch.save(book_embeddings, embeddings_path)
    print(f"Saved book embeddings to {embeddings_path}.")


def get_recommendations(combined_csv_path, user_description, top_n=10):
    # Ask the user to describe their desired book
    # Generate a vector embedding for the user's query
    # Search against the pre-computed embeddings using cosine similarity
    # Return the top matches
    df = pd.read_csv(combined_csv_path)
    query_embedding = model.encode(user_description, convert_to_tensor=True)
    book_embeddings = torch.load('book_embeddings.pt')
    similarities = util.pytorch_cos_sim(query_embedding, book_embeddings)[0]
    similarities = similarities.cpu()
    top_results = similarities.topk(k=top_n)
    similar_books = df.iloc[top_results.indices]['Book_Title'].tolist()
    return similar_books


if __name__ == "__main__":
    download()
    combine_csv_files('data/tanguypledel_science-fiction-books-subgenres', 'combined_science_fiction_books.csv')
    generate_embeddings('combined_science_fiction_books.csv')
    while True:
        user_query = input("\nPlease enter a description of a book that you would like to read: ")
        similar_books = get_recommendations("combined_science_fiction_books.csv", user_query, 10)
        print("Top ten recommendations:")
        for i, book in enumerate(similar_books, 1):
            print(f"{i}. {book}")