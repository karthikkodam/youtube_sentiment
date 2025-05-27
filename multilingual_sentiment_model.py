import os
os.environ["USE_TF"] = "0"

import os
import re
import logging
import textwrap
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from googleapiclient.discovery import build
from transformers import pipeline
from dotenv import load_dotenv
from transformers import pipeline
import streamlit as st

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

@st.cache_resource
def load_sentiment_model():
    from transformers import pipeline
    return pipeline(model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", top_k=None)

sentiment_classifier = load_sentiment_model()

def extract_video_id(url):
    patterns = [
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^?]+)",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([^?]+)",
        r"(?:https?:\/\/)?youtu\.be\/([^?]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_title(video_id):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    try:
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        return response["items"][0]["snippet"]["title"]
    except:
        return "Unknown Title"

def get_comments(video_id, max_results=100):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments = []
    next_page_token = None
    try:
        while len(comments) < max_results:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_results - len(comments)),
                textFormat="plainText",
                pageToken=next_page_token
            )
            response = request.execute()
            for item in response.get("items", []):
                comments.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
    except Exception as e:
        return [], str(e)
    return comments, None

def analyze_sentiment(comments):
    results = []
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    all_sentiments = sentiment_classifier(comments, batch_size=8)
    for comment, scores in zip(comments, all_sentiments):
        sentiment = max(scores, key=lambda x: x["score"])
        label = sentiment["label"]
        results.append({"Comment": comment, "Sentiment": label, "Score": sentiment["score"]})
        counts[label] += 1
    return results, counts

def plot_pie_chart(counts, video_title):
    labels = list(counts.keys())
    values = list(counts.values())
    # colors = ['#66bb6a', '#ffee58', '#ef5350']

    fig = px.pie(
        names=labels,
        values=values,
        title=f"Sentiment Distribution",
        color=labels,
        color_discrete_map={
            'Positive': '#66bb6a',
            'Neutral': '#ffee58',
            'Negative': '#ef5350'
        },
    )
    fig.update_traces(textinfo='percent+label', textfont_size=14)
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
        width=350,
        showlegend=False,
        title_x=0
    )
    return fig

def get_overall_sentiment(counts):
    return f"Overall Video Sentiment: {max(counts, key=counts.get).upper()}"