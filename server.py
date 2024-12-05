import os
import pandas as pd
import uvicorn

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from book_recommendation import get_recommendations

load_dotenv()
app = FastAPI()

class FetchBookRecommendationsRequest(BaseModel):
    query: str

@app.post("/fetchBookRecommendations")
def app_get_book_recommendation(request: FetchBookRecommendationsRequest):
    combined_csv_path = 'combined_science_fiction_books.csv'
    top_n = 10
    recommendations = get_recommendations(combined_csv_path, request.query, top_n)
    return {
        "recommendations": recommendations
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)