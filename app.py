import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# NLTK setup
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def scrape_reviews_robust(amazon_url, api_key):
    # We use ScraperAPI to bypass Amazon's blocks
    payload = {'api_key': 794cbc9f707ab541b2ab889156d16e8d, 'url': amazon_url}
    
    try:
        # This sends the request through a proxy that handles CAPTCHAs
        response = requests.get('http://api.scraperapi.com', params=payload, timeout=60)
        
        if response.status_code != 200:
            return None, f"Error: Received status code {response.status_code}"
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Updated selectors for 2026 Amazon layout
        review_elements = soup.select('span[data-hook="review-body"]')
        
        reviews = [item.get_text().strip() for item in review_elements]
        return reviews, None
    except Exception as e:
        return None, str(e)

# --- UI ---
st.set_page_config(page_title="Amazon AI Dashboard", layout="wide")
st.title("🚀 Amazon Insight Dashboard")

# Sidebar for API Key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter ScraperAPI Key:", type="password", help="Get a free key at scraperapi.com")
    st.info("Amazon blocks standard requests. A proxy API is required for cloud deployment.")

target_url = st.text_input("Amazon Review Page URL:")

if st.button("Run Analysis"):
    if not api_key:
        st.error("Please enter a ScraperAPI key in the sidebar.")
    elif not target_url:
        st.warning("Please enter a URL.")
    else:
        with st.spinner("Bypassing Amazon security..."):
            reviews, error = scrape_reviews_robust(target_url, api_key)
            
            if error:
                st.error(f"Failed: {error}")
            elif not reviews:
                st.warning("Found 0 reviews. Try the URL of the 'All Reviews' page.")
            else:
                # Analysis Logic
                sia = SentimentIntensityAnalyzer()
                data = []
                for r in reviews:
                    score = sia.polarity_scores(r)['compound']
                    sentiment = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
                    data.append({"Review": r[:200] + "...", "Sentiment": sentiment, "Score": score})
                
                df = pd.DataFrame(data)
                
                # Dashboard display
                c1, c2 = st.columns(2)
                c1.metric("Total Reviews Found", len(df))
                c2.metric("Avg Sentiment", f"{df['Score'].mean():.2f}")
                
                st.bar_chart(df['Sentiment'].value_counts())
                st.dataframe(df)
