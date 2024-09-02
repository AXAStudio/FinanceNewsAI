from flask import Flask
from .resources import api as ticker_sentiment_analysis_ns
from .resources import apib as bias_analysis_ns
from .extensions import api


sem_ver = 1.0
api_prefix = f"/api/{sem_ver}"


def create_app() -> Flask:
    """Create the Flask app"""
    app = Flask(__name__)

    api.init_app(app, title="Analytics API")

    api.add_namespace(ticker_sentiment_analysis_ns, path=f"{api_prefix}/sentiment")
    api.add_namespace(bias_analysis_ns, path=f"{api_prefix}/bias")

    return app