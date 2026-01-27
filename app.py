import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download necessary NLTK data
nltk.download('vader_lexicon')

## --- Backend Functions ---

def scrape_reviews(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    
    reviews = []
    # Simplified selector - Amazon often changes these classes
    review_elements = soup.find_all("span", {"data-hook": "review-body"})
    
    for item in review_elements:
        reviews.append(item.get_text().strip())
    return reviews

def analyze_sentiment(review_list):
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for text in review_list:
        score = analyzer.polarity_scores(text)
        if score['compound'] >= 0.05:
            sentiment = "Positive"
        elif score['compound'] <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        results.append({"Review": text, "Sentiment": sentiment, "Score": score['compound']})
    return pd.DataFrame(results)

## --- Streamlit Dashboard UI ---

st.set_page_config(page_title="Amazon Review Insight", layout="wide")
st.title("📊 Amazon Product Review Dashboard")

product_url = st.text_input("Enter Amazon Product Review Page URL:", 
                            placeholder="https://www.amazon.com/product-reviews/ASIN_HERE/...")

if st.button("Analyze Reviews"):
    if product_url:
        with st.spinner("Fetching reviews..."):
            # 1. Scrape
            raw_reviews = scrape_reviews(product_url)
            
            if not raw_reviews:
                st.error("Could not fetch reviews. Amazon may be blocking the request. Try a different URL or a Proxy API.")
            else:
                # 2. Analyze
                df = analyze_sentiment(raw_reviews)
                
                # 3. Present Dashboard
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sentiment Distribution")
                    st.bar_chart(df['Sentiment'].value_counts())
                
                with col2:
                    st.subheader("Key Stats")
                    avg_score = df['Score'].mean()
                    st.metric("Average Sentiment Score", f"{avg_score:.2f}")
                    st.metric("Total Reviews Analyzed", len(df))

                st.divider()
                st.subheader("Detailed Review Analysis")
                st.dataframe(df, use_container_width=True)
    else:
        st.warning("Please enter a URL first.")
