import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import torch
from transformers import pipeline
from bertopic import BERTopic
from umap import UMAP
import spacy
from tqdm import tqdm
import json
import os
import random
import traceback
import logging

# ===================== CONFIGURATION SECTION =====================

# File paths
INPUT_CSV = 'ingestion/raw_files/review_db.csv'
OUTPUT_DIR = "ingestion/processed_destinations"

# Processing settings
CHUNK_SIZE = 10000  # Adjust based on your system's memory capacity
PERCENTAGE_TO_PROCESS = 10  # Percentage of data to process (0.0 to 100.0)

# NLP model configurations
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
SPACY_MODEL = "en_core_web_sm"

# BERTopic settings
BERTOPIC_LANGUAGE = "english"
BERTOPIC_CALCULATE_PROBABILITIES = True
BERTOPIC_VERBOSE = True
UMAP_N_COMPONENTS = 2  # Reduced from 5
UMAP_N_NEIGHBORS = 3  # Reduced from 15
UMAP_MIN_DIST = 0.0
UMAP_METRIC = 'cosine'

# Analysis settings
MIN_REVIEWS_FOR_TOPIC_MODELING = 5  # Minimum number of reviews required for topic modeling
SENTIMENT_SAMPLE_SIZE = 100  # Limit for sentiment analysis
TOPIC_MODELING_SAMPLE_SIZE = 100  # Reduced from 1000 to handle smaller datasets
TOP_ENTITIES_LIMIT = 10
SAMPLE_REVIEWS_LIMIT = 3

# Seasonal analysis
SEASONS = ['Winter', 'Spring', 'Summer', 'Autumn']

# Entity types of interest
ENTITY_TYPES = ['FAC', 'ORG', 'GPE', 'LOC']

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== END OF CONFIGURATION SECTION =====================

# Set up the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load NLP models
sentiment_analyzer = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device=device)
nlp = spacy.load(SPACY_MODEL)

# Initialize UMAP with desired parameters
umap_model = UMAP(n_neighbors=UMAP_N_NEIGHBORS, 
                  n_components=UMAP_N_COMPONENTS, 
                  min_dist=UMAP_MIN_DIST, 
                  metric=UMAP_METRIC,
                  random_state=42)

# Initialize BERTopic with the UMAP model
topic_model = BERTopic(
    language=BERTOPIC_LANGUAGE,
    calculate_probabilities=BERTOPIC_CALCULATE_PROBABILITIES,
    verbose=BERTOPIC_VERBOSE,
    umap_model=umap_model
)

# Initialize data structures to accumulate results
accumulated_data = defaultdict(lambda: {
    'reviews': [],
    'ratings': [],
    'seasons': [],
    'entities': defaultdict(int)
})

def process_chunk(chunk):
    for _, row in chunk.iterrows():
        key = (row['City'], row['Place'])
        accumulated_data[key]['reviews'].append(row['Review'])
        accumulated_data[key]['ratings'].append(row['Rating'])
        
        # Handle date and season
        try:
            parsed_date = datetime.strptime(row['Date'], '%Y-%m-%d').date()
            season = SEASONS[parsed_date.month % 12 // 3]
        except (ValueError, TypeError):
            # If date is invalid or missing, assign a random season
            season = random.choice(SEASONS)
        
        accumulated_data[key]['seasons'].append(season)
        
        # Process entities
        doc = nlp(row['Review'])
        for ent in doc.ents:
            if ent.label_ in ENTITY_TYPES:
                accumulated_data[key]['entities'][ent.text] += 1

def analyze_sentiment(texts):
    results = sentiment_analyzer(texts[:SENTIMENT_SAMPLE_SIZE])
    return sum(result['score'] * (1 if result['label'] == 'POSITIVE' else -1) for result in results) / len(results)

def create_place_profile(key, data):
    city, place = key
    reviews = data['reviews']
    ratings = data['ratings']
    seasons = data['seasons']

    # Basic stats
    avg_rating = np.mean(ratings)
    try:
        sentiment_score = analyze_sentiment(reviews)
    except Exception as e:
        logging.error(f"Error in sentiment analysis for {city}, {place}: {str(e)}")
        sentiment_score = None

    # Seasonal analysis
    seasonal_ratings = {season: [] for season in SEASONS}
    for season, rating in zip(seasons, ratings):
        seasonal_ratings[season].append(rating)
    seasonal_avg = {season: np.mean(ratings) for season, ratings in seasonal_ratings.items() if ratings}
    best_season = max(seasonal_avg, key=seasonal_avg.get) if seasonal_avg else None

    # Topic modeling
    if len(reviews) >= MIN_REVIEWS_FOR_TOPIC_MODELING:
        sample_size = min(TOPIC_MODELING_SAMPLE_SIZE, len(reviews))
        sampled_reviews = np.random.choice(reviews, sample_size, replace=False)
        try:
            topics, _ = topic_model.fit_transform(sampled_reviews)
            top_topics = topic_model.get_topic_info().head(5)
            categories = [
                {"id": f"cat_{i}", "name": topic, "score": 1.0 - (i * 0.1)}
                for i, topic in enumerate(top_topics['Name'].tolist()) if topic != -1
            ]
            if not categories:
                categories = [{"id": "cat_0", "name": "general", "score": 1.0}]
        except Exception as e:
            logging.error(f"Error in topic modeling for {city}, {place}: {str(e)}")
            categories = [{"id": "cat_0", "name": "general", "score": 1.0}]
    else:
        categories = [{"id": "cat_0", "name": "general", "score": 1.0}]

    # Entity extraction
    top_entities = sorted(data['entities'].items(), key=lambda x: x[1], reverse=True)[:TOP_ENTITIES_LIMIT]

    profile = {
        "id": f"place_{city}_{place}".replace(" ", "_").lower(),
        "name": place,
        "city": city,
        "overall_rating": float(avg_rating),
        "sentiment_score": float(sentiment_score),
        "review_count": len(data['reviews']),  # Use original count
        "seasonal_data": [
            {"season": season, "rating": float(rating), "popular_activities": []}
            for season, rating in seasonal_avg.items()
        ],
        "best_season": best_season,
        "categories": categories,
        "top_entities": [
            {"id": f"entity_{i}", "name": entity, "type": "attraction", "count": count}
            for i, (entity, count) in enumerate(top_entities)
        ],
        "sample_reviews": [
            {
                "id": f"review_{i}",
                "text": review[:200],
                "sentiment": sentiment_analyzer(review[:200])[0]['label'],
                "topics": [],
                "mentioned_entities": []
            } for i, review in enumerate(np.random.choice(data['reviews'], min(SAMPLE_REVIEWS_LIMIT, len(data['reviews'])), replace=False))
        ],
        "relationships": {
            "located_in": city,
            "near_attractions": [],
            "similar_to": []
        },
        "geo": {
            "latitude": None,
            "longitude": None
        }
    }
    
    return profile

def main():
    total_records = sum(1 for _ in open(INPUT_CSV)) - 1  # Count total records, subtract 1 for header
    processed_records = 0
    skipped_places = 0
    error_count = 0
    
    chunks_to_process = int((total_records // CHUNK_SIZE) * (PERCENTAGE_TO_PROCESS / 100))
    target_records = min(chunks_to_process * CHUNK_SIZE, total_records)
    
    for chunk in tqdm(pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE), 
                      total=chunks_to_process, 
                      desc="Processing chunks"):
        try:
            process_chunk(chunk)
        except Exception as e:
            error_count += 1
            logging.error(f"Error processing chunk: {str(e)}\n{traceback.format_exc()}")
            continue  # Move to the next chunk
        
        for key, data in accumulated_data.items():
            if len(data['reviews']) >= MIN_REVIEWS_FOR_TOPIC_MODELING:
                try:
                    profile = create_place_profile(key, data)
                    city, place = key
                    filename = f"{OUTPUT_DIR}/{city}_{place.replace(' ', '_')}.json"
                    
                    with open(filename, 'w') as f:
                        json.dump(profile, f, indent=2)
                except Exception as e:
                    error_count += 1
                    logging.error(f"Error creating profile for {key}: {str(e)}\n{traceback.format_exc()}")
                    continue  # Move to the next place
            else:
                skipped_places += 1
        
        processed_records += len(chunk)
        print(f"Processed {processed_records:,} out of {target_records:,} target records. "
              f"{target_records - processed_records:,} target records remaining.")
        print(f"Skipped {skipped_places} places due to insufficient reviews. Errors encountered: {error_count}")
        
        accumulated_data.clear()

        if processed_records >= target_records:
            break

    print(f"Finished processing {processed_records:,} records ({PERCENTAGE_TO_PROCESS}% of total).")
    print(f"Total places skipped due to insufficient reviews: {skipped_places}")
    print(f"Total errors encountered: {error_count}")
    print(f"Check data_processing.log for error details.")

if __name__ == "__main__":
    main()