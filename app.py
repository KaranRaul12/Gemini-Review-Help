import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# 1. Initialize NLTK Sentiment Engine
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def scrape_amazon_reviews(url):
    """Fetches reviews using ScraperAPI and Streamlit Secrets."""
    # Retrieve the key you saved in the 'Secrets' tab
    try:
        api_key = st.secrets["SCRAPER_API_KEY"]
    except KeyError:
        st.error("🔑 API Key not found! Go to Settings > Secrets and add SCRAPER_API_KEY = 'your_key_here'")
        st.stop()

    payload = {'api_key': api_key, 'url': url}
    
    try:
        # Proxy request to bypass Amazon blocks
        response = requests.get('http://api.scraperapi.com', params=payload, timeout=60)
        
        if response.status_code != 200:
            return None, f"ScraperAPI Error: {response.status_code}"
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Scrapes the review text blocks
        review_elements = soup.select('span[data-hook="review-body"]')
        
        if not review_elements:
            return None, "No reviews found. Make sure you use a 'See All Reviews' URL."
            
        reviews = [item.get_text().strip() for item in review_elements]
        return reviews, None
    except Exception as e:
        return None, str(e)

# --- DASHBOARD UI ---
st.set_page_config(page_title="Amazon Review Analyst", layout="wide")

st.title("📊 Amazon Review Dashboard")
st.markdown("Analyze customer sentiment instantly using AI.")

# Input Section
target_url = st.text_input("Enter Amazon Review URL:", 
                           placeholder="https://www.amazon.com/product-reviews/B0XXX...")

if st.button("Analyze Product"):
    if not target_url:
        st.warning("Please paste a URL first.")
    else:
        # Clean URL: ensure https
        if not target_url.startswith("http"):
            target_url = "https://" + target_url

        with st.spinner("🕵️‍♂️ Bypassing Amazon security and analyzing text..."):
            reviews, error = scrape_amazon_reviews(target_url)
            
            if error:
                st.error(f"❌ {error}")
            else:
                # 2. Sentiment Analysis Logic
                sia = SentimentIntensityAnalyzer()
                processed_data = []
                
                for r in reviews:
                    scores = sia.polarity_scores(r)
                    compound = scores['compound']
                    
                    if compound >= 0.05:
                        sentiment = "Positive"
                    elif compound <= -0.05:
                        sentiment = "Negative"
                    else:
                        sentiment = "Neutral"
                        
                    processed_data.append({
                        "Review": r[:300] + "...", 
                        "Sentiment": sentiment, 
                        "Score": compound
                    })
                
                df = pd.DataFrame(processed_data)

                # 3. Dashboard Layout
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Reviews", len(df))
                col2.metric("Positive Reviews", len(df[df['Sentiment'] == 'Positive']))
                col3.metric("Avg Score", f"{df['Score'].mean():.2f}")

                st.divider()

                # Visual Charts
                chart_col, table_col = st.columns([1, 1])
                
                with chart_col:
                    st.subheader("Sentiment Share")
                    st.bar_chart(df['Sentiment'].value_counts())
                
                with table_col:
                    st.subheader("Data Summary")
                    st.dataframe(df[['Sentiment', 'Score']], use_container_width=True)

                st.divider()
                st.subheader("Full Review Analysis")
                st.table(df)

                # 4. Download Option
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Analysis as CSV", data=csv, file_name="amazon_analysis.csv", mime="text/csv")
