import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import nltk
import re

# Initialize NLTK
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- Custom Styling ---
st.set_page_config(page_title="Amazon Insight Pro", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .sentiment-pos { color: #28a745; font-weight: bold; }
    .sentiment-neg { color: #dc3545; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

def scrape_amazon_reviews(url):
    try:
        api_key = st.secrets["SCRAPER_API_KEY"]
    except KeyError:
        st.error("🔑 API Key missing in Secrets!")
        st.stop()

    payload = {'api_key': api_key, 'url': url, 'country_code': 'us'}
    try:
        response = requests.get('http://api.scraperapi.com', params=payload, timeout=60)
        soup = BeautifulSoup(response.text, "html.parser")
        review_elements = soup.select('span[data-hook="review-body"]')
        return [item.get_text().strip() for item in review_elements], None
    except Exception as e:
        return None, str(e)

def get_keywords(text_list):
    # Simple cleaner to find common nouns/adjectives
    all_text = " ".join(text_list).lower()
    words = re.findall(r'\w+', all_text)
    # Filter out common stop words
    stop_words = {'the', 'and', 'i', 'it', 'to', 'is', 'was', 'of', 'for', 'in', 'with', 'this', 'but', 'on', 'my'}
    words = [w for w in words if w not in stop_words and len(w) > 3]
    return Counter(words).most_common(10)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=100)
    st.title("Settings")
    st.info("Ensure you use the 'See All Reviews' page URL for the best results.")
    target_url = st.text_input("Amazon Review URL:")
    analyze_btn = st.button("🚀 Run Full Analysis", use_container_width=True)

# --- MAIN DASHBOARD ---
st.title("📦 Product Sentiment Intelligence")

if analyze_btn and target_url:
    if not target_url.startswith("http"):
        target_url = "https://" + target_url

    with st.spinner("Analyzing customer feedback..."):
        reviews, error = scrape_amazon_reviews(target_url)
        
        if error:
            st.error(f"Error: {error}")
        elif not reviews:
            st.warning("No reviews found. Amazon might be showing a CAPTCHA or the URL structure is different.")
        else:
            # Data Processing
            sia = SentimentIntensityAnalyzer()
            data = []
            for r in reviews:
                score = sia.polarity_scores(r)['compound']
                sentiment = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
                data.append({"Review": r, "Sentiment": sentiment, "Score": score})
            
            df = pd.DataFrame(data)
            
            # --- TOP ROW: KPI METRICS ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Reviews", len(df))
            m2.metric("Positive %", f"{(len(df[df['Sentiment']=='Positive'])/len(df)*100):.1f}%")
            m3.metric("Negative %", f"{(len(df[df['Sentiment']=='Negative'])/len(df)*100):.1f}%")
            m4.metric("Avg Score", f"{df['Score'].mean():.2f}")

            st.divider()

            # --- MIDDLE ROW: CHARTS ---
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("Sentiment Distribution")
                st.bar_chart(df['Sentiment'].value_counts(), color="#FF9900")
            
            with c2:
                st.subheader("Trending Keywords")
                keywords = get_keywords(reviews)
                key_df = pd.DataFrame(keywords, columns=['Word', 'Count'])
                st.dataframe(key_df, use_container_width=True, hide_index=True)

            # --- BOTTOM ROW: DETAILED TABLE ---
            st.subheader("Customer Voices")
            
            # Highlight sentiments in the table
            def highlight_sentiment(val):
                color = '#d4edda' if val == 'Positive' else '#f8d7da' if val == 'Negative' else '#e2e3e5'
                return f'background-color: {color}'

            st.dataframe(df.style.applymap(highlight_sentiment, subset=['Sentiment']), use_container_width=True)

            # Export
            st.download_button("Export Results", df.to_csv().encode('utf-8'), "analysis.csv", "text/csv")
else:
    st.image("https://img.freepik.com/free-vector/sentiment-analysis-concept-illustration_114360-5182.jpg", width=400)
    st.write("Enter a URL in the sidebar and click 'Run Analysis' to begin.")
