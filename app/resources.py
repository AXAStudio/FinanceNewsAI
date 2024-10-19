import os
from flask import request
from flask_restx import Resource, Namespace
import asyncio
from .sentiment import SentimentAnalysis
from .api_models import analyze_model, bias_model
from .bias import get_bias

import tensorflow as tf


DEFAULT_ARTICLE_COUNT = 10


model = tf.keras.models.load_model('app/models/sentiment_model.keras')
api = Namespace("Sentiment Analysis")
apib = Namespace("Bias Analysis")


@api.route('/ticker')
class TickerSentimentAPI(Resource):
    """Set up returns and sentiment API"""
    @api.expect(analyze_model)
    @api.response(200, "Successfully Retrieved Sentiment Analysis")
    @api.response(400, "Failed to Retrieve Sentiment Analysis")
    def post(self):
        """
        Uses the sentiment analysis model to get news articles and predict the 
        sentiment with the sentiment AI
        """
        try:
            data = request.json
            asset = data.get('asset')
            max_articles = data.get('max_articles', DEFAULT_ARTICLE_COUNT)
            
            sentiment_analysis = SentimentAnalysis(
                asset,
                model,
                max_articles=max_articles,
                requires_ticker=True
            )
            result = asyncio.run(sentiment_analysis.analyze_sentiment())
            return {'success': 1, 'body': result}, 200
        except Exception as e:
            return {'success': 0, 'body': str(e)}, 400
        
@api.route('/company')
class CompanySentimentAPI(Resource):
    """Set up returns and sentiment API"""
    @api.expect(analyze_model)
    @api.response(200, "Successfully Retrieved Sentiment Analysis")
    @api.response(400, "Failed to Retrieve Sentiment Analysis")
    def post(self):
        """
        Uses the sentiment analysis model to get news articles and predict the 
        sentiment with the sentiment AI
        """
        try:
            data = request.json
            asset = data.get('asset')
            max_articles = data.get('max_articles', DEFAULT_ARTICLE_COUNT)
            
            sentiment_analysis = SentimentAnalysis(
                asset,
                model,
                max_articles=max_articles,
                requires_ticker=False
            )
            result = asyncio.run(sentiment_analysis.analyze_sentiment())
            return {'success': 1, 'body': result}, 200
        except Exception as e:
            return {'success': 0, 'body': str(e)}, 400
        
@api.route('/general')
class GeneralSentimentAPI(Resource):
    """Set up returns and sentiment API"""
    @api.expect(analyze_model)
    @api.response(200, "Successfully Retrieved Sentiment Analysis")
    @api.response(400, "Failed to Retrieve Sentiment Analysis")
    def post(self):
        """
        Uses the sentiment analysis model to get news articles and predict the 
        sentiment with the sentiment AI
        """
        try:
            data = request.json
            asset = data.get('asset')
            max_articles = data.get('max_articles', DEFAULT_ARTICLE_COUNT)
            
            sentiment_analysis = SentimentAnalysis(
                asset,
                model,
                max_articles=max_articles,
                requires_ticker=False
            )
            result = asyncio.run(sentiment_analysis.analyze_sentiment())
            return {'success': 1, 'body': result}, 200
        except Exception as e:
            return {'success': 0, 'body': str(e)}, 400
        

@apib.route('')
class BiasAPI(Resource):
    """Set up returns and bias API"""
    @apib.expect(bias_model)
    @apib.response(200, "Successfully Retrieved Bias Analysis")
    @apib.response(400, "Failed to Retrieve Bias Analysis")
    def post(self):
        """
        Processes input data to perform a bias analysis and returns the result 
        with a success or error status.
        """
        try:
            data = request.json
            outlet = data.get('outlet')
            result = get_bias(outlet)
            return {'success': 1, 'body': result}, 200
        except Exception as e:
            return {'success': 0, 'body': str(e)}, 400

