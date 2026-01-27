import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
from google import genai
import nltk

# Initialize NLTK for sentiment analysis
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
    .stButton>button { border-radius: 10px; border: 1px solid #FF9900; }
    </style>
    """, unsafe_allow_html=True)

# --- AI LOGIC ---

def get_ai_response(user_query, reviews_context):
    """Uses the 2026 Google GenAI SDK with automatic model fallback."""
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        
        # Priority list for 2026 models
        model_candidates = ["gemini-3-flash", "gemini-3-flash-preview", "gemini-2.5-flash"]
        
        prompt = f"""
        You are a product analyst. Based ONLY on the Amazon reviews provided below, answer the question.
        QUESTION: {user_query}
        REVIEWS: {str(reviews_context)[:15000]}
        """

        # Try models in order until one works
        for model_name in model_candidates:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text
            except Exception:
                continue
                
        return "Error: No compatible Gemini models found. Check your API quota."
    except Exception as e:
        return f"Connection Error: {str(e)}"

def scrape_amazon(url):
    """Scrapes Amazon reviews via ScraperAPI."""
    try:
        api_key = st.secrets["SCRAPER_API_KEY"]
        payload = {'api_key': api_key, 'url': url}
        response = requests.get('http://api.scraperapi.com', params=payload, timeout=60)
        soup = BeautifulSoup(response.text, "html.parser")
        review_elements = soup.select('span[data-hook="review-body"]')
        return [item.get_text().strip() for item in review_elements], None
    except Exception as e:
        return None, str(e)

# --- SESSION STATE ---
if 'reviews_list' not in st.session_state:
    st.session_state.reviews_list = []
if 'chat_answer' not in st.session_state:
    st.session_state.chat_answer = ""

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Controls")
    target_url = st.text_input("Amazon 'See All Reviews' URL:", placeholder="Paste link here...")
    
    if st.button("🔍 Analyze Product", use_container_width=True):
        if target_url:
            with st.spinner("Scraping and analyzing sentiment..."):
                reviews, error = scrape_amazon(target_url)
                if reviews:
                    st.session_state.reviews_list = reviews
                    st.session_state.chat_answer = "" 
                    st.success(f"Loaded {len(reviews)} reviews!")
                else:
                    st.error(f"Scrape Failed: {error}")

# --- MAIN DASHBOARD ---
st.title("📊 Amazon Customer Intelligence Dashboard")

if st.session_state.reviews_list:
    reviews = st.session_state.reviews_list
    sia = SentimentIntensityAnalyzer()
    
    # Data Processing
    df = pd.DataFrame([{"Review": r, "Score": sia.polarity_scores(r)['compound']} for r in reviews])
    df['Sentiment'] = df['Score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))

    # Visuals
    col_metrics, col_chart = st.columns([1, 2])
    
    with col_metrics:
        st.metric("Total Reviews Scraped", len(df))
        pos_perc = (len(df[df['Sentiment']=='Positive'])/len(df))*100
        st.metric("Positive Sentiment", f"{pos_perc:.1f}%")
        st.metric("Avg Sentiment Score", f"{df['Score'].mean():.2f}")

    with col_chart:
        fig = px.histogram(df, x='Score', nbins=20, title="Sentiment Score Distribution",
                           color_discrete_sequence=['#FF9900'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Chatbot UI
    st.subheader("💬 AI Review Analyst")
    st.info("Ask questions about the reviews, like 'Is the packaging good?' or 'What do people hate?'")
    
    c1, c2, c3 = st.columns([2, 1, 1])
    user_input = c1.text_input("Enter your question:", key="user_query")
    
    # Quick action buttons
    if c2.button("Summarize Top Pros"): 
        st.session_state.chat_answer = get_ai_response("What are the top 3 strengths of this product?", reviews)
    if c3.button("Summarize Top Cons"): 
        st.session_state.chat_answer = get_ai_response("What are the top 3 weaknesses or complaints?", reviews)

    if user_input:
        with st.spinner("AI is thinking..."):
            st.session_state.chat_answer = get_ai_response(user_input, reviews)

    if st.session_state.chat_answer:
        st.markdown(f'<div class="chat-box"><b>AI Analyst:</b><br>{st.session_state.chat_answer}</div>', unsafe_allow_html=True)

    # Raw Data
    st.divider()
    st.subheader("Raw Review Data")
    st.dataframe(df, use_container_width=True)

else:
    st.info("👈 Please enter an Amazon Review URL in the sidebar to start.")
