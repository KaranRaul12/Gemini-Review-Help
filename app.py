import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import google.generativeai as genai
import nltk

# Initialize NLTK
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Amazon Insight AI", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    [data-testid="stMetricValue"] { color: #FF9900 !important; }
    .chat-container { background-color: #1f2937; padding: 20px; border-radius: 15px; border: 1px solid #374151; }
    </style>
    """, unsafe_allow_html=True)

# --- BACKEND FUNCTIONS ---

def scrape_amazon(url):
    api_key = st.secrets["SCRAPER_API_KEY"]
    payload = {'api_key': api_key, 'url': url}
    try:
        response = requests.get('http://api.scraperapi.com', params=payload, timeout=60)
        soup = BeautifulSoup(response.text, "html.parser")
        review_elements = soup.select('span[data-hook="review-body"]')
        return [item.get_text().strip() for item in review_elements], None
    except Exception as e:
        return None, str(e)

def get_ai_response(user_query, reviews_context):
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are a product analyst. Based on these Amazon reviews, answer the user's question.
    REVIEWS: {str(reviews_context)[:10000]} 
    QUESTION: {user_query}
    """
    response = model.generate_content(prompt)
    return response.text

# --- DASHBOARD UI ---
st.title("📊 Amazon Insight + AI Chat")

if 'reviews_list' not in st.session_store:
    st.session_store.reviews_list = []

with st.sidebar:
    st.header("🛒 Setup")
    target_url = st.text_input("Amazon Review URL:")
    if st.button("🚀 Analyze Product", use_container_width=True):
        with st.spinner("Scraping and analyzing..."):
            reviews, error = scrape_amazon(target_url)
            if reviews:
                st.session_store.reviews_list = reviews
                st.success(f"Found {len(reviews)} reviews!")
            else:
                st.error(error)

# --- ANALYSIS SECTION ---
if st.session_store.reviews_list:
    reviews = st.session_store.reviews_list
    sia = SentimentIntensityAnalyzer()
    df = pd.DataFrame([{"Review": r, "Score": sia.polarity_scores(r)['compound']} for r in reviews])
    df['Sentiment'] = df['Score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Sentiment Distribution")
        fig = px.pie(df, names='Sentiment', color='Sentiment',
                     color_discrete_map={'Positive':'#FF9900', 'Negative':'#FF0000', 'Neutral':'#808080'},
                     hole=0.4)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # --- CHATBOT SECTION ---
    with col2:
        st.subheader("💬 Ask the AI about these reviews")
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        user_input = st.text_input("e.g., 'What are the top 3 pros and cons?'")
        
        if user_input:
            with st.spinner("AI is thinking..."):
                answer = get_ai_response(user_input, reviews)
                st.markdown(f"**AI:** {answer}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Detailed Breakdown")
    
    def style_sent(val):
        color = '#006400' if val == 'Positive' else '#8b0000' if val == 'Negative' else '#444444'
        return f'background-color: {color}; color: white; font-weight: bold;'
    
    st.dataframe(df.style.map(style_sent, subset=['Sentiment']), use_container_width=True)

else:
    st.info("👈 Enter an Amazon Review URL in the sidebar to start.")
