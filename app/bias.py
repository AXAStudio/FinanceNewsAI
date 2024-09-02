import requests
from bs4 import BeautifulSoup
import re

def get_bias(news_outlet: str) -> dict:
    """Pull bias from Media Bias/Fact Check (MBFC) and return cleaned bias info."""
    news_outlet = news_outlet.lower()
    base_url = 'https://mediabiasfactcheck.com/?s='
    search_url = f'{base_url}{news_outlet.replace(" ", "+")}'
    
    try:
        response = requests.get(search_url)
        response.raise_for_status()
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}
    
    soup = BeautifulSoup(response.text, 'html.parser')
    search_results = soup.find_all('h3', class_='entry-title')
    
    if not search_results:
        return {"bias": f"No results found for '{news_outlet}'."}
    
    outlet_url = search_results[0].find('a')['href']
    
    try:
        response = requests.get(outlet_url)
        response.raise_for_status()
    except requests.RequestException as e:
        return {"error": f"Request to outlet URL failed: {e}"}
    
    soup = BeautifulSoup(response.text, 'html.parser')
    bias_section = soup.find('div', class_='entry-content')
    
    if not bias_section:
        return {"bias": f"Bias and reliability information for '{news_outlet}' not found."}
    
    bias_info = {}
    info_labels = {
        'Bias Rating:': 'bias',
        'Factual Reporting:': 'factual_reporting',
        'Country:': 'country',
        'MBFC’s Country Freedom Rating:': 'freedom_rating',
        'Media Type:': 'media_type',
        'Traffic/Popularity:': 'traffic_popularity',
        'MBFC Credibility Rating:': 'credibility_rating'
    }

    text = bias_section.get_text(" ", strip=True)
    for label, key in info_labels.items():
        pattern = re.compile(f"{label} (.*?)(?= {'|'.join(info_labels.keys())})", re.DOTALL)
        match = pattern.search(text)
        if match:
            if ": " in match:
                match = match.split(": ")[1]
            bias_info[key] = match.group(1).strip()

    if bias_info:
        return bias_info
    
    return {"bias": f"Bias and reliability information for '{news_outlet}' not fully found."}
