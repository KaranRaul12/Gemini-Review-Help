import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import nltk
import re

# Initialize NLTK
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- UI & DARK MODE STYLING ---
st.set_page_config(page_title="Amazon Insight Pro", layout="wide")

# Custom CSS to fix "white panes" and contrast
st.markdown("""
    <style>
    /* Main background */
    .stApp { background-color: #0e1117; color: white; }
    
    /* KPI Card Styling */
    [data-testid="stMetricValue"] { color: #FF9900 !important; font-size: 2rem; }
    [data-testid="stMetricLabel"] { color: #ffffff !important; }
    div[data-testid="metric-container"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def scrape_amazon_reviews(url):
    try:
        api_key = st.secrets["SCRAPER_API_KEY"]
    except KeyError:
        st.error("🔑 API Key missing in Secrets!")
        st.stop()

    payload = {'api_key': api_key, 'url': url}
    try:
        response = requests.get('http://api.scraperapi.com', params=payload, timeout=60)
        soup = BeautifulSoup(response.text, "html.parser")
        review_elements = soup.select('span[data-hook="review-body"]')
        return [item.get_text().strip() for item in review_elements], None
    except Exception as e:
        return None, str(e)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Settings")
    target_url = st.text_input("Amazon Review URL:")
    analyze_btn = st.button("🚀 Run Analysis", use_container_width=True)

# --- MAIN DASHBOARD ---
st.title("📦 Product Sentiment Intelligence")

if analyze_btn and target_url:
    with st.spinner("Fetching Data..."):
        reviews, error = scrape_amazon_reviews(target_url)
        
        if error:
            st.error(f"Error: {error}")
        elif not reviews:
            st.warning("No reviews found. Try the 'See All Reviews' page URL.")
        else:
            # 1. Processing Logic
            sia = SentimentIntensityAnalyzer()
            data = []
            for r in reviews:
                score = sia.polarity_scores(r)['compound']
                sentiment = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
                data.append({"Review": r, "Sentiment": sentiment, "Score": score})
            
            df = pd.DataFrame(data)
            
            # --- ROW 1: KPI METRICS ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Reviews", len(df))
            m2.metric("Positive Reviews", len(df[df['Sentiment']=='Positive']))
            m3.metric("Negative Reviews", len(df[df['Sentiment']=='Negative']))
            m4.metric("Avg Score", f"{df['Score'].mean():.2f}")

            st.divider()

            # --- ROW 2: PIE CHART & KEYWORDS ---
            c1, c2 = st.columns([1, 1])
            
            with c1:
                st.subheader("Sentiment Share")
                # Custom Pie Chart: Orange for Positive, Red for Negative
                fig = px.pie(df, names='Sentiment', 
                             color='Sentiment',
                             color_discrete_map={'Positive':'#FF9900', 'Negative':'#FF0000', 'Neutral':'#808080'},
                             hole=0.4)
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                st.subheader("Top Feedback Keywords")
                words = re.findall(r'\w+', " ".join(reviews).lower())
                stop_words = {'the', 'and', 'was', 'for', 'this', 'with', 'that', 'they'}
                filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
                key_df = pd.DataFrame(pd.Series(filtered_words).value_counts().head(8)).reset_index()
                key_df.columns = ['Keyword', 'Frequency']
                st.table(key_df)

            # --- ROW 3: STYLED DATA TABLE ---
            st.subheader("Detailed Review Breakdown")
            
            # This function fixes the "white washed" look with bold colors
            def style_sentiment(val):
                if val == 'Positive':
                    return 'background-color: #006400; color: white; font-weight: bold;' # Dark Green
                elif val == 'Negative':
                    return 'background-color: #8b0000; color: white; font-weight: bold;' # Dark Red
                return 'background-color: #444444; color: white;' # Grey

            # Applying the style to the Sentiment column
            styled_df = df.style.map(style_sentiment, subset=['Sentiment'])
            st.dataframe(styled_df, use_container_width=True)

            st.download_button("📥 Download Report", df.to_csv().encode('utf-8'), "amazon_report.csv")
else:
    st.info("👈 Paste an Amazon 'See All Reviews' URL in the sidebar to begin.")
