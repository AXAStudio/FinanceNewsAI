# Senteco AI

Senteco AI is an AI-powered financial news sentiment analysis system.  
It uses a deep learning model (BiLSTM + TensorFlow) to process market news headlines in real time, returning structured sentiment scores for companies, tickers, and assets.

---

## Features

- **Finance-Specific Sentiment Model**  
  Trained on a large dataset of financial news headlines for bullish/bearish classification.

- **Live News Retrieval**  
  Pulls the latest headlines from Google News RSS, filtered by company name or ticker symbol.

- **Structured API Output**  
  Returns the average sentiment score, individual article scores, publication dates, sources, and links.

- **Robust Evaluation**  
  Uses Stratified K-Fold Cross Validation to ensure consistent model performance.

- **REST API Implementation**  
  Built with Flask and asyncio for low-latency, concurrent requests.

---

## Example Output

```json
{
  "asset_details": {
    "asset_name": "Amazon",
    "asset_ticker": "AMZN"
  },
  "n_articles_found": 10,
  "avg_score": 0.42,
  "oldest_article_read": "3 Days Ago",
  "data": {
    "0": {
      "headline": "Amazon shares rise after strong earnings report",
      "cover": "https://www.projectactionstar.com/uploads/videos/no_image.gif",
      "score": 0.87,
      "date": "Thu, 08 Aug 2025 13:00:00 GMT",
      "outlet": "Reuters",
      "article_links": "https://www.reuters.com/article/amazon-earnings"
    }
  }
}
