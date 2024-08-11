from flask import Flask, request, jsonify
import aiohttp
import asyncio
from yahooquery import search
from bs4 import BeautifulSoup
import tensorflow as tf
import time
from bias import get_bias

app = Flask(__name__)

class SentimentAnalysis:
    def __init__(self, asset, model):
        self.asset = asset
        self.model = model
        self.placeholder_url = "https://www.projectactionstar.com/uploads/videos/no_image.gif"
    
    async def fetch_company_info(self):
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
    
    def listdict(self, keys, values, keyval):        
        return {
            str(i): {keyval[j]: values[j][i] for j in range(len(keyval))}
            for i in range(len(keys))
        }
    
    async def fetch_news_page(self, session, url):
        async with session.get(url) as response:
            if response.status != 200:
                return None
            return await response.text()
    
    async def analyze_sentiment(self):
        asset, valid_ticker = await self.fetch_company_info() if self.asset else (None, False)
        if not asset:
            return {'status': 404, 'message': 'could not fetch name'}

        asset = asset.title()
        all_news, all_images, all_time, all_outlets, all_bias, articlelinks = [], [], [], [], [], []
        num_pages = 5

        async with aiohttp.ClientSession() as session:
            tasks = [
                self.fetch_news_page(session, f'https://news.search.yahoo.com/search?p={asset}&b={page * 10 + 1}')
                for page in range(num_pages)
            ]
            pages = await asyncio.gather(*tasks)

        for page_content in pages:
            if not page_content:
                continue
            soup = BeautifulSoup(page_content, 'html.parser')
            for news_item in soup.find_all('div', class_='NewsArticle'):
                title = news_item.find('h4').text.strip()
                outlet = news_item.find('span', class_='s-source mr-5 cite-co').text.strip()
                articlelink = news_item.find('a')['href']
                if outlet.lower() == "benzinga":
                    outlet = "Benzinga via AOL"
                timetext = news_item.find('span', class_='fc-2nd').text.replace('·', '').strip()
                bias = get_bias(outlet)
                all_news.append(title)
                all_outlets.append(outlet)
                all_time.append(timetext)
                all_bias.append(bias)
                articlelinks.append(articlelink)
                all_images.extend(self.fetch_image_links(soup.findAll('img')))
                if len(all_news) >= 20:
                    break
            if len(all_news) >= 20:
                break

        all_news = list(dict.fromkeys(all_news))[:20]
        all_time = all_time[:len(all_news)]
        all_images = all_images[:len(all_news)]
        all_outlets = all_outlets[:len(all_news)]
        all_bias = all_bias[:len(all_news)]
        articlelinks = articlelinks[:len(all_news)]

        # Pad images, time, and outlets if they are shorter than all_news
        all_images += [self.placeholder_url] * (len(all_news) - len(all_images))
        all_time += ["Unknown time"] * (len(all_news) - len(all_time))
        all_outlets += ["Unknown Outlet"] * (len(all_news) - len(all_outlets))

        newsindex = list(range(len(all_news)))

        try:
            predictions = self.model(tf.constant(all_news)).numpy().tolist()
            avg = sum((prediction[0] if prediction[0] > 0.5 else -1 * (1 - prediction[0])) for prediction in predictions) / len(predictions)
            scores = [float('%.3f' % (prediction[0] if prediction[0] > 0.5 else -1 * (1 - prediction[0]))) for prediction in predictions]

            # Make sure scores length matches the length of all_news
            scores += [0.0] * (len(all_news) - len(scores))

            data = self.listdict(newsindex, [all_news, all_images, scores[:len(all_news)], all_time, all_outlets, all_bias, articlelinks], ["headline", "cover", "score", "age", "outlet", "outlet_bias", "article_links"])
            response = {
                'status': 200,
                'avg_score': float('%.3f' % avg),
                'data': data
            }
        except Exception as e:
            print(f"Error during sentiment analysis: {e}")
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
    result = asyncio.run(sentiment_analysis.analyze_sentiment())
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
