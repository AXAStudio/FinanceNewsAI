import os
import aiohttp
import asyncio
from typing import Union, Any
from yahooquery import search
from bs4 import BeautifulSoup
import math


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf


ARTICLES_PER_PAGE = 7


class TFModel:
    """Saved tensorflow model"""
    def __init__(self, model: Any):
        """Constructor"""
        self._model = model

    def predict(self, model_input: list) -> list:
        """Get predictions from model"""
        return self._model(tf.constant(model_input)).numpy().tolist()


class SentimentAnalysis:
    """Gets sentiment analysis for an asset using a saved model"""
    def __init__(
        self,
        asset: str, 
        model: Any,
        max_articles: int = 10,
        default_image_url: str = "https://www.projectactionstar.com/uploads\
/videos/no_image.gif",
        requires_ticker: bool = False
    ) -> "SentimentAnalysis":
        """Constructor"""
        self.asset = asset
        self.model = TFModel(model)
        self.max_articles = max_articles
        self.placeholder_url = default_image_url
        self._requiresTicker = requires_ticker
    
    async def fetch_company_info(
        self
    ) -> tuple[Union[None, str], Union[None, str]]:
        """
        If company_name is a ticker, converts to asset_name otherwise, leaves 
        blank
        """
        company_name = self.asset.upper()
        try:
            results = search(company_name)
            quotes = results['quotes'][0]
            return quotes['longname'], quotes['symbol']
        except Exception:
            if (self._requiresTicker):
                return None, None
            else:
                return self.asset, None

    def fetch_image_links(self, images: str) -> list[str]:
        """
        Gets the cover page of a specific article and replaces none with 
        placeholder
        """
        return [
            image.get(attr) if "logo" not in (image.get(attr) or "").lower() 
            and "data:image" not in (image.get(attr) or "")
            else self.placeholder_url
            for image in images
            for attr in ["dimg_2", "data-src", "data-fallback-src", "src"]
            if image.get(attr)
        ]

    def listdict(
        self,
        keys: list,
        keyval: list, 
        values: list  
    ) -> dict[str, dict[str, Union[str, int]]]:
        """
        Takes three lists and outputs format {listone:{listtwo: listthree}}
        """    
        return {
            str(i): {keyval[j]: values[j][i] for j in range(len(keyval))}
            for i in range(len(keys))
        }

    async def fetch_news_page(
        self, 
        session: aiohttp.ClientSession, 
        url: str
    ) -> str:

        """Fetches html on web page"""
        async with session.get(url) as response:
            if response.status != 200:
                return None
            return await response.text()
    
    async def analyze_sentiment(self) -> dict[str]:
        """
        Gets news headlines and feeds them into the Sentiment AI which outputs a
        score from -1 to 1
        """

        asset_name, ticker = await self.fetch_company_info()

        if not asset_name:
            return {"error": f"Could not find asset: {self.asset}"}
        
        asset_name = asset_name.replace(" ", "+")
        asset_name = asset_name.title()

        url = f"https://news.google.com/rss/search?q={asset_name}"

        async with aiohttp.ClientSession() as session:
            rss_feed = await self.fetch_news_page(session, url)

        if not rss_feed:
            return {"error": "Failed to retrieve the news feed."}

        soup = BeautifulSoup(rss_feed, "xml")

        items = soup.find_all("item")[:self.max_articles]
        all_news = [item.title.text for item in items]

        all_outlets = [
            item.source.text 
            if item.source else "Unknown Outlet" 
            for item in items
        ]
        
        all_time = [item.pubDate.text for item in items]
        articlelinks = [item.link.text for item in items]
        all_images = [self.placeholder_url] * len(all_news)
        
        if not all_news:
            return {"error": "No news articles found"}

        predictions = self.model.predict(all_news)

        def pred_convert_binary(pred: int) -> int:
            """Convert 0 - 1 to -1 to 1"""
            return pred if pred > 0.5 else -1 * (1 - pred)

        avg = sum(
            pred_convert_binary(prediction[0])
            for prediction in predictions
        ) / len(predictions)

        scores = [
            float('%.3f' % pred_convert_binary(prediction[0]))
            for prediction in predictions
        ]

        scores += [0.0] * (len(all_news) - len(scores))

        asset_name = asset_name.replace("+"," ")

        newsindex = list(range(len(all_news)))

        listandheaders = {
            "headline": all_news,
            "cover": all_images,
            "score": scores[:len(all_news)],
            "date": all_time,
            "outlet": all_outlets,
            "article_links": articlelinks
        }

        data = self.listdict(
            newsindex, 
            list(listandheaders.keys()),
            list(listandheaders.values())
        )

        asset_details = {
            'asset_name': asset_name,
            'asset_ticker': ticker
        }

        response = {
            'asset_details': asset_details,
            'n_articles_found': len(all_news),
            'avg_score': float('%.3f' % avg),
            'data': data
        }

        return response