import requests
from bs4 import BeautifulSoup

def get_bias(news_outlet):
    news_outlet = news_outlet.lower()  # Lowercase the input
    base_url = 'https://mediabiasfactcheck.com/?s='
    search_url = f'{base_url}{news_outlet.replace(" ", "+")}'
    
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Attempt to locate the search result and navigate to the first result's page
    search_results = soup.find_all('h3', class_='entry-title')
    if not search_results:
        return {"error": f"No results found for '{news_outlet}'."}
    
    outlet_url = search_results[0].find('a')['href']
    
    # Visit the outlet's page to get bias and reliability data
    response = requests.get(outlet_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extracting the bias and reliability information
    bias_section = soup.find('div', class_='entry-content')
    
    if bias_section:
        paragraphs = bias_section.find_all('p')
        bias_info = {}
        
        for paragraph in paragraphs:
            text = paragraph.text.strip()
            if 'Bias Rating:' in text:
                bias_info['bias'] = text.split("Bias Rating:")[-1].split('\n')[0].strip()
            if 'Factual Reporting:' in text:
                bias_info['factual_reporting'] = text.split("Factual Reporting:")[-1].split('\n')[0].strip()
            if 'Country:' in text:
                bias_info['country'] = text.split("Country:")[-1].split('\n')[0].strip()
            if 'MBFC’s Country Freedom Rating:' in text:
                bias_info['freedom_rating'] = text.split("MBFC’s Country Freedom Rating:")[-1].split('\n')[0].strip()
            if 'Media Type:' in text:
                bias_info['media_type'] = text.split("Media Type:")[-1].split('\n')[0].strip()
            if 'Traffic/Popularity:' in text:
                bias_info['traffic_popularity'] = text.split("Traffic/Popularity:")[-1].split('\n')[0].strip()
            if 'MBFC Credibility Rating:' in text:
                bias_info['credibility_rating'] = text.split("MBFC Credibility Rating:")[-1].split('\n')[0].strip()
        
        if bias_info:
            return bias_info
        else:
            return {"error": f"Bias and reliability information for '{news_outlet}' not fully found."}
    else:
        return {"error": f"Bias and reliability information for '{news_outlet}' not found."}

    

