import kagglehub
import os
import pandas as pd
import shutil
import torch
from itertools import combinations
from jaccard_similarity import average_jaccard_similarity
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
    torch.save(book_embeddings.to("cpu"), embeddings_path)
    print(f"Saved book embeddings to {embeddings_path}.")

def calculate_diversity(recommendations, embeddings, df):
    # Determine how diverse the recommended book suggestions are
    # Generate similarity scores for every pair among the recommendations
    # Calculate the diversity score by subtracting the average simlarity from one
    indices = []
    for book in recommendations:
        if book in df['Book_Title'].values:
            index = df[df['Book_Title'] == book].index[0]
            indices.append(index)
    book_embeddings = embeddings[indices]
    book_pair_similarities = []
    for (i, j) in combinations(range(len(book_embeddings)), 2):
        similarity = util.pytorch_cos_sim(book_embeddings[i], book_embeddings[j]).item()
        book_pair_similarities.append(similarity)
    if book_pair_similarities:
        average_similarity = sum(book_pair_similarities) / len(book_pair_similarities)
    else:
        average_similarity = 0
    diversity_score = 1 - average_similarity
    return diversity_score

def get_recommendations(combined_csv_path, user_description, top_n=10):
    # Ask the user to describe their desired book
    # Generate a vector embedding for the user's query
    # Search against the pre-computed embeddings using cosine similarity
    # Return the top matches
    df = pd.read_csv(combined_csv_path)
    query_embedding = model.encode(user_description, convert_to_tensor=True).to('cpu')
    # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    book_embeddings = torch.load('book_embeddings.pt', map_location=torch.device("cpu"))
    similarities = util.pytorch_cos_sim(query_embedding, book_embeddings)[0]
    similarities = similarities.cpu()
    top_results = similarities.topk(k=top_n)
    similar_books = df.iloc[top_results.indices]['Book_Title'].tolist()
    return similar_books, book_embeddings, df

if __name__ == "__main__":
    if not os.path.exists('data/tanguypledel_science-fiction-books-subgenres'):
        download()
    if not os.path.exists('combined_science_fiction_books.csv'):
        combine_csv_files('data/tanguypledel_science-fiction-books-subgenres', 'combined_science_fiction_books.csv')
    if not os.path.exists('book_embeddings.pt'):
        generate_embeddings('combined_science_fiction_books.csv')
    while True:
        user_query = input("\nPlease enter a description of a book that you would like to read or type in q and press enter to exit: ")
        if user_query.lower() == 'q':
            print("Goodbye!")
            break
        similar_books, book_embeddings, df = get_recommendations("combined_science_fiction_books.csv", user_query, 10)
        avg_jaccard_similarity = average_jaccard_similarity(similar_books)
        diversity_score = calculate_diversity(similar_books, book_embeddings, df)
        print("Top ten recommendations:")
        for i, book in enumerate(similar_books, 1):
            print(f"{i}. {book}")
        print(f"Jaccard Similarity: {avg_jaccard_similarity:.4f}")
        print(f"Diversity score: {diversity_score:.4f}")