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

# --- UI CONFIG & STYLING ---
st.set_page_config(page_title="Amazon AI Insights", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    [data-testid="stMetricValue"] { color: #FF9900 !important; font-weight: bold; }
    .chat-box { background-color: #1f2937; padding: 20px; border-radius: 15px; border: 1px solid #FF9900; }
    .stButton>button { border-radius: 8px; font-weight: bold; }
    /* Horizontal Print Fix */
    @media print { .stSidebar { display: none !important; } }
    </style>
    """, unsafe_allow_html=True)

# --- AI CORE ---
def get_ai_response(user_query, reviews_context):
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        prompt = f"Product Analyst: {user_query}\n\nREVIEWS:\n{str(reviews_context)[:12000]}"
        # Using stable 2.5 series to avoid quota/not found errors
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

def scrape_amazon(url):
    try:
        api_key = st.secrets["SCRAPER_API_KEY"]
        payload = {'api_key': api_key, 'url': url}
        response = requests.get('http://api.scraperapi.com', params=payload, timeout=60)
        soup = BeautifulSoup(response.text, "html.parser")
        return [el.get_text().strip() for el in soup.select('span[data-hook="review-body"]')], None
    except Exception as e:
        return None, str(e)

# --- SESSION ---
if 'reviews_list' not in st.session_state: st.session_state.reviews_list = []
if 'chat_answer' not in st.session_state: st.session_state.chat_answer = ""

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=140)
    st.markdown("---")
    target_url = st.text_input("Amazon Review URL:")
    if st.button("🔍 Run Full Analysis", use_container_width=True, type="primary"):
        if target_url:
            with st.spinner("Scraping..."):
                reviews, error = scrape_amazon(target_url)
                if reviews:
                    st.session_state.reviews_list = reviews
                    st.session_state.chat_answer = ""
                else: st.error(error)

    if st.session_state.reviews_list:
        st.markdown("---")
        st.button("🖨️ Print Report", on_click=lambda: st.write('<script>window.print();</script>', unsafe_allow_html=True))
        df_csv = pd.DataFrame(st.session_state.reviews_list, columns=["Reviews"])
        st.download_button("📥 Download CSV", df_csv.to_csv(index=False), "reviews.csv", "text/csv", use_container_width=True)

# --- MAIN PAGE ---
st.title("📦 Amazon Customer Intelligence Dashboard")

if st.session_state.reviews_list:
    reviews = st.session_state.reviews_list
    sia = SentimentIntensityAnalyzer()
    df = pd.DataFrame([{"Review": r, "Score": sia.polarity_scores(r)['compound']} for r in reviews])
    df['Sentiment'] = df['Score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))

    # Visuals Row
    col_pie, col_bar = st.columns(2)
    with col_pie:
        fig_pie = px.pie(df, names='Sentiment', color='Sentiment', title="Sentiment Share",
                         color_discrete_map={'Positive':'#1b5e20', 'Negative':'#b71c1c', 'Neutral':'#FF9900'}, hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_bar:
        fig_bar = px.histogram(df, x='Score', title="Sentiment Intensity", color_discrete_sequence=['#FF9900'])
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # Chatbot with Quick Actions
    st.subheader("💬 AI Review Analyst")
    
    # Predefined Quick-Query Buttons
    col_q1, col_q2, col_q3 = st.columns([1, 1, 2])
    if col_q1.button("✅ Top 3 Pros", use_container_width=True):
        st.session_state.chat_answer = get_ai_response("What are the top 3 specific pros mentioned by users?", reviews)
    if col_q2.button("❌ Top 3 Cons", use_container_width=True):
        st.session_state.chat_answer = get_ai_response("What are the top 3 recurring complaints or cons?", reviews)
    
    user_query = st.text_input("Ask a custom question:", placeholder="e.g. Is the battery life good?")
    if user_query:
        with st.spinner("AI is analyzing..."):
            st.session_state.chat_answer = get_ai_response(user_query, reviews)

    if st.session_state.chat_answer:
        st.markdown(f'<div class="chat-box"><b>AI Response:</b><br>{st.session_state.chat_answer}</div>', unsafe_allow_html=True)

    st.divider()

    # Colored Explorer
    st.subheader("📝 Review Data Explorer")
    def color_sentiment(val):
        if val == 'Positive': return 'background-color: #1b5e20; color: white'
        if val == 'Negative': return 'background-color: #b71c1c; color: white'
        return 'background-color: #FF9900; color: black'
    
    st.dataframe(df.style.applymap(color_sentiment, subset=['Sentiment']), use_container_width=True)
else:
    st.info("👈 Please enter a URL in the sidebar to begin analysis.")
