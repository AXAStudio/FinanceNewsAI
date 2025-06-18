import os
import aiohttp
import asyncio
from typing import Union, Any
from yahooquery import search
from bs4 import BeautifulSoup
from datetime import datetime
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

maxlen = 100  # consistent with training

def preprocess_text_list(texts: list[str], tokenizer, maxlen: int = 100):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

class TFModel:
    def __init__(self, model: Any):
        self._model = model

    def predict(self, model_input: np.ndarray) -> np.ndarray:
        return self._model.predict(model_input)

class SentimentAnalysis:
    def __init__(
        self,
        asset: str,
        model: Any,
        max_articles: int = 10,
        default_image_url: str = "https://www.projectactionstar.com/uploads/videos/no_image.gif",
        requires_ticker: bool = False
    ):
        self.asset = asset
        self.model = TFModel(model)
        self.max_articles = max_articles
        self.placeholder_url = default_image_url
        self._requiresTicker = requires_ticker

    async def fetch_company_info(self) -> tuple[Union[None, str], Union[None, str]]:
        try:
            results = search(self.asset.upper())
            quotes = results['quotes'][0]
            return quotes['longname'], quotes['symbol']
        except Exception:
            return (None, None) if self._requiresTicker else (self.asset, None)

    async def fetch_news_page(self, session: aiohttp.ClientSession, url: str) -> str:
        async with session.get(url) as response:
            return await response.text() if response.status == 200 else None

    def listdict(self, keys, keyval, values):
        return {
            str(i): {keyval[j]: values[j][i] for j in range(len(keyval))}
            for i in range(len(keys))
        }

    def find_oldest_article(self, data: dict) -> str:
        dates = [datetime.strptime(article["date"], "%a, %d %b %Y %H:%M:%S %Z") for article in data.values()]
        if not dates:
            return None
        days = (datetime.now() - min(dates)).days
        return f"{days} Days Ago"

    async def analyze_sentiment(self) -> dict[str, Any]:
        asset_name, ticker = await self.fetch_company_info()
        if not asset_name:
            return {"error": f"Could not find asset: {self.asset}"}
        
        asset_query = asset_name.replace(" ", "+")
        url = f"https://news.google.com/rss/search?q={asset_query}"

        async with aiohttp.ClientSession() as session:
            rss_feed = await self.fetch_news_page(session, url)
        if not rss_feed:
            return {"error": "Failed to retrieve the news feed."}

        soup = BeautifulSoup(rss_feed, "xml")
        items = soup.find_all("item")[:self.max_articles]
        all_news = [item.title.text for item in items]
        all_outlets = [item.source.text if item.source else "Unknown Outlet" for item in items]
        all_time = [item.pubDate.text for item in items]
        articlelinks = [item.link.text for item in items]
        all_images = [self.placeholder_url] * len(all_news)

        if not all_news:
            return {"error": "No news articles found"}

        preprocessed = preprocess_text_list(all_news, tokenizer, maxlen)
        predictions = self.model.predict(preprocessed)

        scores = [(pred[0] - 0.5) * 2 for pred in predictions]  # Normalize to -1 to 1
        scores = [round(score, 3) for score in scores]
        avg_score = round(sum(scores) / len(scores), 3)

        # Output structuring
        newsindex = list(range(len(all_news)))
        listandheaders = {
            "headline": all_news,
            "cover": all_images,
            "score": scores,
            "date": all_time,
            "outlet": all_outlets,
            "article_links": articlelinks
        }
        data = self.listdict(newsindex, list(listandheaders.keys()), list(listandheaders.values()))
        days_oldest = self.find_oldest_article(data)

        return {
            'asset_details': {'asset_name': asset_name.replace("+", " "), 'asset_ticker': ticker},
            'n_articles_found': len(all_news),
            'avg_score': avg_score,
            'oldest_article_read': days_oldest,
            'data': data
        }
