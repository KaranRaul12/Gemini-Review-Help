import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
from google import genai
import nltk

# Initialize NLTK
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- UI CONFIG ---
st.set_page_config(page_title="Amazon Insight AI 2026", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    [data-testid="stMetricValue"] { color: #FF9900 !important; }
    .chat-box { background-color: #1f2937; padding: 20px; border-radius: 15px; border: 1px solid #374151; margin-top: 10px; }
    .stButton>button { border-radius: 10px; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

# --- AI & SCRAPING LOGIC ---

def get_ai_response(user_query, reviews_context):
    """Uses the new google-genai SDK for Gemini 3 Flash."""
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        prompt = f"Analyze these reviews and answer: {user_query}\n\nREVIEWS:\n{str(reviews_context)[:15000]}"
        
        # Using gemini-3-flash (Stable as of Jan 2026)
        response = client.models.generate_content(
            model="gemini-3-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}. Check your API key or model availability."

def scrape_amazon(url):
    try:
        api_key = st.secrets["SCRAPER_API_KEY"]
        payload = {'api_key': api_key, 'url': url}
        response = requests.get('http://api.scraperapi.com', params=payload, timeout=60)
        soup = BeautifulSoup(response.text, "html.parser")
        review_elements = soup.select('span[data-hook="review-body"]')
        return [item.get_text().strip() for item in review_elements], None
    except Exception as e:
        return None, str(e)

# --- SESSION INITIALIZATION ---
if 'reviews_list' not in st.session_state:
    st.session_state.reviews_list = []
if 'chat_answer' not in st.session_state:
    st.session_state.chat_answer = ""

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Dashboard Controls")
    target_url = st.text_input("Paste Amazon Review URL:", placeholder="https://amazon.com/...")
    if st.button("🔍 Run Full Analysis", use_container_width=True):
        if target_url:
            with st.spinner("Scraping reviews..."):
                reviews, error = scrape_amazon(target_url)
                if reviews:
                    st.session_state.reviews_list = reviews
                    st.session_state.chat_answer = "" # Reset chat for new product
                else:
                    st.error(f"Scrape Failed: {error}")

# --- MAIN DASHBOARD ---
st.title("📦 Amazon Customer Intelligence")

if st.session_state.reviews_list:
    reviews = st.session_state.reviews_list
    sia = SentimentIntensityAnalyzer()
    
    # Process Data
    df = pd.DataFrame([{"Review": r, "Score": sia.polarity_scores(r)['compound']} for r in reviews])
    df['Sentiment'] = df['Score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))

    # Metrics Row
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Reviews", len(df))
    m2.metric("Positive Share", f"{(len(df[df['Sentiment']=='Positive'])/len(df)*100):.1f}%")
    m3.metric("Avg Score", f"{df['Score'].mean():.2f}")

    st.divider()

    # Chat & Viz Row
    col_viz, col_chat = st.columns([1, 1.2])

    with col_viz:
        st.subheader("Sentiment Distribution")
        fig = px.pie(df, names='Sentiment', color='Sentiment',
                     color_discrete_map={'Positive':'#FF9900', 'Negative':'#FF0000', 'Neutral':'#636E72'},
                     hole=0.5)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_chat:
        st.subheader("💬 Chat with Reviews")
        # Quick prompts
        c1, c2 = st.columns(2)
        q1 = c1.button("What are the Pros?")
        q2 = c2.button("Main Complaints?")
        
        user_input = st.text_input("Ask a question:", value="What's the general consensus?" if q1 or q2 else "")
        if q1: user_input = "What are the 3 best things about this product?"
        if q2: user_input = "What are the top 3 recurring complaints in these reviews?"

        if user_input:
            with st.spinner("AI analyzing reviews..."):
                st.session_state.chat_answer = get_ai_response(user_input, reviews)
        
        if st.session_state.chat_answer:
            st.markdown(f'<div class="chat-box"><b>AI Response:</b><br>{st.session_state.chat_answer}</div>', unsafe_allow_html=True)

    # Detailed Table
    st.divider()
    st.subheader("Review Data Explorer")
    def color_sentiment(val):
        color = '#1b5e20' if val == 'Positive' else '#b71c1c' if val == 'Negative' else '#424242'
        return f'background-color: {color}; color: white; font-weight: bold'

    st.dataframe(df.style.map(color_sentiment, subset=['Sentiment']), use_container_width=True)

else:
    st.info("👋 Welcome! Please enter an Amazon URL in the sidebar to begin analysis.")
