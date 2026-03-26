import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from google import genai
import nltk

# Initialize NLTK
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- UI CONFIG & STYLING (DARK FRAMEWORK) ---
st.set_page_config(page_title="SENTIMENT ANALYSIS", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;600&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 50% 50%, #12141d 0%, #050505 100%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    .gradient-text {
        background: linear-gradient(92deg, #FF9900 0%, #FF5F6D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Orbitron', sans-serif;
        font-size: 2.8rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 20px;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        backdrop-filter: blur(10px);
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .stats-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
    }
    .stats-table td {
        padding: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stats-label { font-weight: 600; color: #e0e0e0; }
    .stats-value { color: #FF9900; font-weight: bold; text-align: right; font-family: 'Orbitron'; }

    .chat-box {
        background: linear-gradient(145deg, rgba(28,31,43,1) 0%, rgba(14,17,23,1) 100%);
        border: 1px solid #333;
        border-left: 4px solid #FF9900;
        padding: 25px;
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- IMPROVED METADATA LOGIC ---
def get_product_metadata(reviews, title):
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        # We give the AI the Title and a small snippet of reviews for context
        context_text = f"Title: {title} | Review Snippet: {str(reviews)[:1000]}"
        
        prompt = (
            "You are a product expert. Based ONLY on the context provided, identify the: "
            "1. Manufacturing Company, 2. Product Model Name, 3. Product Category. "
            "Format your response EXACTLY like this: Company | Model | Category. "
            "If you cannot find one, use 'Unknown'. Do not write anything else."
        )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=[prompt, context_text]
        )
        
        parts = response.text.split('|')
        if len(parts) == 3:
            return [p.strip() for p in parts]
        return ["Unknown", "Unknown", "Unknown"]
    except Exception as e:
        return ["Error", "Error", "Error"]

def get_stats_data(reviews):
    dimensions = {
        'Quality': ['quality', 'build', 'premium', 'material'],
        'Value': ['price', 'worth', 'money', 'value'],
        'Usability': ['easy', 'use', 'setup', 'friendly'],
        'Durability': ['last', 'sturdy', 'strong', 'broke'],
        'Service': ['shipping', 'package', 'customer', 'delivery']
    }
    sia = SentimentIntensityAnalyzer()
    scores = []
    for dim, keywords in dimensions.items():
        rel = [r for r in reviews if any(k in r.lower() for k in keywords)]
        avg = sum([sia.polarity_scores(r)['compound'] for r in rel])/len(rel) if rel else 0
        scores.append(round(((avg + 1) / 2) * 10, 1))
    return list(dimensions.keys()), scores

def get_ai_response(query, context):
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        res = client.models.generate_content(model="gemini-2.0-flash", contents=f"Analyst. Context: {str(context)[:8000]}. Q: {query}")
        return res.text
    except Exception as e: return f"AI Error: {str(e)}"

def scrape_amazon(url):
    try:
        api_key = st.secrets["SCRAPER_API_KEY"]
        payload = {'api_key': api_key, 'url': url, 'render': 'true'}
        res = requests.get('http://api.scraperapi.com', params=payload, timeout=60)
        if res.status_code != 200: return None, None, f"Error {res.status_code}"
        soup = BeautifulSoup(res.text, "html.parser")
        
        # Priority on finding the title for metadata
        title = soup.find("span", {"id": "productTitle"}) or soup.find("h1", {"id": "title"})
        title_text = title.get_text().strip() if title else "Unknown Product"
        
        revs = [el.get_text().strip() for el in soup.select('span[data-hook="review-body"]')]
        if not revs: revs = [el.get_text().strip() for el in soup.select('.review-text-content span')]
        
        return revs, title_text, None
    except Exception as e: return None, None, str(e)

# --- SESSION ---
if 'reviews_list' not in st.session_state: st.session_state.reviews_list = []
if 'meta' not in st.session_state: st.session_state.meta = ["-", "-", "-"]

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=150)
    url = st.text_input("🔗 Paste Amazon Review URL:")
    if st.button("🚀 UNLEASH AI", use_container_width=True):
        if url:
            with st.spinner("Analyzing Product Data..."):
                revs, title, err = scrape_amazon(url)
                if revs:
                    st.session_state.reviews_list = revs
                    # We pass the scraped title here so Gemini has a better chance
                    st.session_state.meta = get_product_metadata(revs, title)
                else: st.error(err)

# --- DASHBOARD MAIN ---
st.markdown('<h1 class="gradient-text">SENTIMENT ANALYSIS</h1>', unsafe_allow_html=True)

if st.session_state.reviews_list:
    reviews = st.session_state.reviews_list
    sia = SentimentIntensityAnalyzer()
    df = pd.DataFrame([{"Review": r, "Score": sia.polarity_scores(r)['compound']} for r in reviews])
    df['Sentiment'] = df['Score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))
    
    avg_score = df['Score'].mean()
    rec, clr = ("MUST BUY", "#00ff88") if avg_score > 0.4 else (("GOOD BUY", "#FF9900") if avg_score > 0.05 else ("THINK AGAIN", "#ff3333"))

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-card"><p style="font-size:0.8rem; opacity:0.7;">COMPANY</p><h3 style="color:#FF9900">{st.session_state.meta[0]}</h3></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><p style="font-size:0.8rem; opacity:0.7;">MODEL</p><h3 style="color:#FF9900">{st.session_state.meta[1]}</h3></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><p style="font-size:0.8rem; opacity:0.7;">CATEGORY</p><h3 style="color:#FF9900">{st.session_state.meta[2]}</h3></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><p style="font-size:0.8rem; opacity:0.7;">RECOMMENDATION</p><h2 style="color:{clr}; font-weight:bold;">{rec}</h2></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- UPDATED: PRODUCT STATISTICS ---
    c_left, c_right = st.columns([1.5, 1])
    labels, values = get_stats_data(reviews)

    with c_left:
        fig_bars = go.Figure(go.Bar(
            x=values, y=labels, orientation='h',
            marker=dict(color='#FF9900'),
            text=[f"{v}/10" for v in values], textposition='auto',
        ))
        fig_bars.update_layout(
            title=dict(text="Product Statistics Graph", font=dict(family="Orbitron", color="#FF9900")),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white",
            xaxis=dict(range=[0, 10], showgrid=False), yaxis=dict(autorange="reversed"),
            height=350, margin=dict(t=50, b=20, l=0, r=20)
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    with c_right:
        st.markdown('<p style="font-family:Orbitron; color:#FF9900; margin-top:10px;">📊 Product Statistics</p>', unsafe_allow_html=True)
        stats_html = '<table class="stats-table">'
        for l, v in zip(labels, values):
            stats_html += f'<tr><td class="stats-label">{l}</td><td class="stats-value">{v}/10</td></tr>'
        stats_html += '</table>'
        st.markdown(stats_html, unsafe_allow_html=True)

    # Neural Analyst
    st.markdown('<h3 style="color:#FF9900; font-family:Orbitron; margin-top:30px;">💬 NEURAL ANALYST</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("✅ Quick Pros"): st.session_state.chat_answer = get_ai_response("Top 3 pros?", reviews)
    if c2.button("❌ Quick Cons"): st.session_state.chat_answer = get_ai_response("Top 3 cons?", reviews)
    user_q = st.text_input("Ask a question about these reviews:")
    if user_q: st.session_state.chat_answer = get_ai_response(user_q, reviews)
    if st.session_state.get('chat_answer'): st.markdown(f'<div class="chat-box">{st.session_state.chat_answer}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)
else:
    st.info("👋 System Standby. Awaiting URL Input.")
