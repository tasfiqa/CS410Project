import pandas as pd
from ast import literal_eval
from itertools import combinations

def get_genres(df):
    df['Genres'] = df['Genres'].apply(literal_eval)
    df['Genres_Unwrapped'] = df['Genres'].apply(lambda genre_dict: list(genre_dict.keys()))
    return df

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0.0

def average_jaccard_similarity(similar_books, df=None):
    if df is None:  
        df = pd.read_csv("combined_science_fiction_books.csv")
    # List of the genres of the books, in order
    genres_list = df[df['Book_Title'].isin(similar_books)]['Genres'].tolist()
    similarities = []
    for genres1, genres2 in combinations(genres_list, 2):
        set1, set2 = set(genres1), set(genres2)
        similarity = jaccard_similarity(set1, set2)
        similarities.append(similarity)
    return sum(similarities) / len(similarities) if similarities else 0.0

if __name__ == "__main__": 
    # Example computing of jaccard similarity for a given set of recommendations
    books = [
        "The Collected Stories of Philip K. Dick 3: Second Variety",
        "The Collected Stories of Philip K. Dick 2: We Can Remember it for You Wholesale",
        "The House on the Strand",
        "The Man in the High Castle",
        "Philip K. Dick's Electric Dreams",
        "The Collected Stories of Philip K. Dick 4: The Minority Report",
        "Dr. Bloodmoney",
        "The Steel Tsar",
        "Radio Free Albemuth",
        "The Collected Stories of Philip K. Dick 1: The Short Happy Life of the Brown Oxford"
    ]
    print(average_jaccard_similarity(books))