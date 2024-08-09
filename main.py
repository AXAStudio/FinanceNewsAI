from flask import Flask, request, jsonify
import requests
from yahooquery import search
from bs4 import BeautifulSoup
import tensorflow as tf
import time

app = Flask(__name__)

class SentimentAnalysis:
    def __init__(self, asset, model):
        self.asset = asset
        self.model = model
        self.placeholder_url = "https://www.projectactionstar.com/uploads/videos/no_image.gif"
    
    def fetch_company_info(self):
        ticker_symbol = self.asset.upper()
        try:
            results = search(ticker_symbol)
            if results['quotes']:
                return results['quotes'][0]['longname'], results['quotes'][0]['symbol'] == ticker_symbol
        except Exception:
            return None, False
        return None, False

    def fetch_image_links(self, images):
        return [
            image.get(attr) if "logo" not in (image.get(attr) or "").lower() and "data:image" not in (image.get(attr) or "")
            else self.placeholder_url
            for image in images
            for attr in ["data-srcset", "data-src", "data-fallback-src", "src"]
            if image.get(attr)
        ]

    def analyze_sentiment(self):
        asset, valid_ticker = self.fetch_company_info() if self.asset else (None, False)
        if not asset:
            return {'status': 404, 'message': 'could not fetch name'}

        asset = asset.title()
        all_news, all_images = [], []
        num_pages = 5

        for page in range(num_pages):
            time.sleep(0.1)
            url = f'https://news.search.yahoo.com/search?p={asset}&b={page * 10 + 1}'
            response = requests.get(url)

            if response.status_code != 200:
                return {'status': 404, 'message': 'could not fetch news'}

            soup = BeautifulSoup(response.text, 'html.parser')
            for news_item in soup.find_all('div', class_='NewsArticle'):
                title = news_item.find('h4').text.strip()
                all_news.append(title)
                all_images.extend(self.fetch_image_links(soup.findAll('img')))
                if len(all_news) >= 20:
                    break
            if len(all_news) >= 20:
                break

        all_news = list(dict.fromkeys(all_news))[:20]
        all_images = list(dict.fromkeys(all_images))[:20]

        try:
            predictions = self.model(tf.constant(all_news)).numpy().tolist()
            avg = sum((prediction[0] if prediction[0] > 0.5 else -1 * (1 - prediction[0])) for prediction in predictions) / len(predictions)
            response = {
                'status': 200,
                'avg_score': float('%.3f' % avg),
                'news': all_news,
                'images': all_images,
                'scores': [float('%.3f' % (prediction[0] if prediction[0] > 0.5 else -1 * (1 - prediction[0]))) for prediction in predictions],
            }
        except Exception:
            return {'status': 400, 'message': 'something went wrong'}

        return response

# Load the TensorFlow model once when the server starts
model = tf.saved_model.load('sentiment_model')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    asset = data.get('asset', None)
    if not asset:
        return jsonify({'status': 400, 'message': 'asset is required'}), 400
    
    sentiment_analysis = SentimentAnalysis(asset, model)
    result = sentiment_analysis.analyze_sentiment()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
