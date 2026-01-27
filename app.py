import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
from google import genai
import nltk
import base64

# Initialize NLTK
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- UI CONFIG & CUSTOM CSS ---
st.set_page_config(page_title="Amazon AI Insights", layout="wide")

# Custom CSS for styling the table and print layout
st.markdown("""
    <style>
    /* Dark Theme Base */
    .stApp { background-color: #0e1117; color: white; }
    
    /* Sentiment Colors for Dataframe */
    .sentiment-pos { background-color: #1b5e20 !important; color: white !important; padding: 5px; border-radius: 5px; }
    .sentiment-neg { background-color: #b71c1c !important; color: white !important; padding: 5px; border-radius: 5px; }
    .sentiment-neu { background-color: #424242 !important; color: white !important; padding: 5px; border-radius: 5px; }

    /* Amazon Theme Elements */
    [data-testid="stMetricValue"] { color: #FF9900 !important; font-weight: bold; }
    .chat-box { background-color: #1f2937; padding: 20px; border-radius: 15px; border: 1px solid #FF9900; }
    
    /* Print Utility */
    @media print {
        .stSidebar, .stButton, header { display: none !important; }
        .stApp { background-color: white !important; color: black !important; }
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def get_ai_response(user_query, reviews_context):
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        prompt = f"Product Analyst: {user_query}\n\nREVIEWS:\n{str(reviews_context)[:12000]}"
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
        reviews = [el.get_text().strip() for el in soup.select('span[data-hook="review-body"]')]
        return reviews, None
    except Exception as e:
        return None, str(e)

# --- SESSION STATE ---
if 'reviews_list' not in st.session_state: st.session_state.reviews_list = []
if 'chat_answer' not in st.session_state: st.session_state.chat_answer = ""

# --- SIDEBAR ---
with st.sidebar:
    # Amazon Logo
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=150)
    st.markdown("---")
    st.header("⚙️ Settings")
    target_url = st.text_input("Amazon Review URL:", placeholder="Paste link here...")
    
    if st.button("🔍 Analyze Product", use_container_width=True, type="primary"):
        if target_url:
            with st.spinner("Processing..."):
                reviews, error = scrape_amazon(target_url)
                if reviews:
                    st.session_state.reviews_list = reviews
                    st.session_state.chat_answer = ""
                else:
                    st.error(error)

    st.markdown("---")
    # Action Buttons
    if st.session_state.reviews_list:
        st.button("🖨️ Print Dashboard", on_click=lambda: st.write('<script>window.print();</script>', unsafe_allow_html=True))
        
        # CSV Export Logic
        df_export = pd.DataFrame(st.session_state.reviews_list, columns=["Review Text"])
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Reviews CSV", data=csv, file_name="amazon_reviews.csv", mime="text/csv", use_container_width=True)

# --- MAIN DASHBOARD ---
st.title("📊 Amazon Customer Intelligence Dashboard")

if st.session_state.reviews_list:
    reviews = st.session_state.reviews_list
    sia = SentimentIntensityAnalyzer()
    df = pd.DataFrame([{"Review": r, "Score": sia.polarity_scores(r)['compound']} for r in reviews])
    df['Sentiment'] = df['Score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))

    # Visual Layout
    col_pie, col_bar = st.columns([1, 1.2])

    with col_pie:
        st.subheader("Sentiment Share")
        # Attractive Pie Chart with custom mapping
        fig_pie = px.pie(df, names='Sentiment', color='Sentiment',
                         color_discrete_map={'Positive':'#1b5e20', 'Negative':'#b71c1c', 'Neutral':'#FF9900'},
                         hole=0.5)
        fig_pie.update_layout(showlegend=True, paper_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        st.subheader("Score Distribution")
        # Attractive Bar Chart (Histogram)
        fig_bar = px.histogram(df, x='Score', nbins=15, 
                               color_discrete_sequence=['#FF9900'], 
                               labels={'Score':'Sentiment Intensity'})
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # AI Section
    st.subheader("💬 Ask AI Analyst")
    user_input = st.text_input("Query the reviews (e.g., 'Is the material high quality?'):")
    if user_input:
        with st.spinner("AI analyzing..."):
            st.session_state.chat_answer = get_ai_response(user_input, reviews)
    
    if st.session_state.chat_answer:
        st.markdown(f'<div class="chat-box"><b>AI Analyst:</b><br>{st.session_state.chat_answer}</div>', unsafe_allow_html=True)

    st.divider()

    # Styled Table
    st.subheader("Review Explorer")
    def color_rows(val):
        if val == 'Positive': return 'background-color: #1b5e20; color: white'
        if val == 'Negative': return 'background-color: #b71c1c; color: white'
        return 'background-color: #FF9900; color: black'

    st.dataframe(df.style.applymap(color_rows, subset=['Sentiment']), use_container_width=True)

else:
    st.info("👋 Enter an Amazon URL in the sidebar to populate the dashboard.")
