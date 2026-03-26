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

# --- UI CONFIG & STYLING ---
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

# --- BACKEND LOGIC ---
def get_product_metadata(reviews, title):
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        context_text = f"Title: {title} | Snippet: {str(reviews)[:1000]}"
        prompt = (
            "Identify: 1. Company, 2. Model, 3. Category. "
            "Return ONLY as: Company | Model | Category."
        )
        response = client.models.generate_content(model="gemini-2.0-flash", contents=[prompt, context_text])
        parts = response.text.split('|')
        return [p.strip() for p in parts] if len(parts) == 3 else ["Unknown", "Unknown", "Unknown"]
    except: return ["Unknown", "Unknown", "Unknown"]

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
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.find("span", {"id": "productTitle"}) or soup.find("h1", {"id": "title"})
        title_text = title.get_text().strip() if title else "Unknown Product"
        revs = [el.get_text().strip() for el in soup.select('span[data-hook="review-body"]')]
        if not revs: revs = [el.get_text().strip() for el in soup.select('.review-text-content span')]
        return revs, title_text, None
    except Exception as e: return None, None, str(e)

# --- APP FLOW ---
if 'reviews_list' not in st.session_state: st.session_state.reviews_list = []
if 'meta' not in st.session_state: st.session_state.meta = ["-", "-", "-"]

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=120)
    url = st.text_input("🔗 Paste Amazon Review URL:")
    if st.button("🚀 UNLEASH AI", use_container_width=True):
        if url:
            with st.spinner("Decoding Product..."):
                revs, title, err = scrape_amazon(url)
                if revs:
                    st.session_state.reviews_list = revs
                    st.session_state.meta = get_product_metadata(revs, title)
                else: st.error(err)

st.markdown('<h1 class="gradient-text">SENTIMENT ANALYSIS</h1>', unsafe_allow_html=True)

if st.session_state.reviews_list:
    reviews = st.session_state.reviews_list
    labels, values = get_stats_data(reviews)
    
    # Metadata Row
    m1, m2, m3, m4 = st.columns(4)
    meta = st.session_state.meta
    avg_score = pd.DataFrame([SentimentIntensityAnalyzer().polarity_scores(r)['compound'] for r in reviews])[0].mean()
    rec, rec_clr = ("MUST BUY", "#00ff88") if avg_score > 0.4 else (("GOOD BUY", "#FF9900") if avg_score > 0.05 else ("THINK AGAIN", "#ff3333"))

    m1.markdown(f'<div class="metric-card"><p style="font-size:0.7rem;">COMPANY</p><h3 style="color:#FF9900">{meta[0]}</h3></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><p style="font-size:0.7rem;">MODEL</p><h3 style="color:#FF9900">{meta[1]}</h3></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><p style="font-size:0.7rem;">CATEGORY</p><h3 style="color:#FF9900">{meta[2]}</h3></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><p style="font-size:0.7rem;">RECOMMENDATION</p><h2 style="color:{rec_clr};">{rec}</h2></div>', unsafe_allow_html=True)

    st.write("<br>", unsafe_allow_html=True)

    # --- UPDATED: DYNAMIC BAR COLORS ---
    c_left, c_right = st.columns([1.5, 1])

    with c_left:
        # Dynamic color logic: Green if >= 5, Yellow if < 5
        bar_colors = ['#00ff88' if v >= 5 else '#FFCC00' for v in values]
        
        fig_bars = go.Figure(go.Bar(
            x=values, y=labels, orientation='h',
            marker=dict(color=bar_colors),
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
    if c1.button("✅ Quick Pros"): st.session_state.chat_ans = get_ai_response("Top 3 pros?", reviews)
    if c2.button("❌ Quick Cons"): st.session_state.chat_ans = get_ai_response("Top 3 cons?", reviews)
    
    user_q = st.text_input("Ask the AI about these reviews:")
    if user_q: st.session_state.chat_ans = get_ai_response(user_q, reviews)
    if st.session_state.get('chat_ans'): st.markdown(f'<div class="chat-box">{st.session_state.chat_ans}</div>', unsafe_allow_html=True)

else:
    st.info("👋 System Standby. Awaiting URL Input.")
