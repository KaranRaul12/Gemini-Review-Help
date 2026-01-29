import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from google import genai
import nltk
from datetime import datetime
import re

# Initialize NLTK
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- UI CONFIG ---
st.set_page_config(page_title="SENTIMENT ANALYSIS", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;600&display=swap');
    .stApp { background: radial-gradient(circle at 50% 50%, #12141d 0%, #050505 100%); color: #e0e0e0; font-family: 'Inter', sans-serif; }
    .gradient-text { background: linear-gradient(92deg, #FF9900 0%, #FF5F6D 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Orbitron', sans-serif; font-size: 2.8rem; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 20px; }
    .metric-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 20px; text-align: center; backdrop-filter: blur(10px); min-height: 110px; display: flex; flex-direction: column; justify-content: center; transition: 0.3s; }
    .metric-card:hover { border-color: #FF9900; transform: translateY(-5px); }
    .chat-box { background: linear-gradient(145deg, rgba(28,31,43,1) 0%, rgba(14,17,23,1) 100%); border-left: 4px solid #FF9900; padding: 25px; border-radius: 15px; margin-top: 20px; }
    .dna-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.85rem; }
    .dna-table td { padding: 6px; border-bottom: 1px solid rgba(255,255,255,0.05); }
    .score-tag { color: #FF9900; font-weight: bold; text-align: right; }
    </style>
    """, unsafe_allow_html=True)

# --- BACKEND LOGIC ---

def parse_amazon_date(date_string):
    try:
        # Extracts "12 January 2024" from "Reviewed in India on 12 January 2024"
        match = re.search(r'(\d+\s+\w+\s+\d{4})', date_string)
        if match:
            return datetime.strptime(match.group(1), '%d %B %Y')
    except:
        return datetime.now()
    return datetime.now()

def scrape_amazon(url):
    try:
        api_key = st.secrets["SCRAPER_API_KEY"]
        res = requests.get('http://api.scraperapi.com', params={'api_key': api_key, 'url': url}, timeout=60)
        soup = BeautifulSoup(res.text, "html.parser")
        
        rev_elements = soup.select('div[data-hook="review"]')
        reviews_data = []
        for el in rev_elements:
            body = el.select_one('span[data-hook="review-body"]')
            date_el = el.select_one('span[data-hook="review-date"]')
            if body and date_el:
                reviews_data.append({
                    'text': body.get_text().strip(),
                    'date': parse_amazon_date(date_el.get_text().strip())
                })
        
        title = soup.find("span", {"id": "productTitle"})
        return reviews_data, (title.get_text().strip() if title else "Product"), None
    except Exception as e: return None, None, str(e)

def get_authenticity_score(reviews):
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        prompt = f"Analyze these reviews for fake patterns. Return: Genuine % | Fake %. Context: {str(reviews)[:4000]}"
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        parts = response.text.split('|')
        gen = int(''.join(filter(str.isdigit, parts[0])))
        fak = int(''.join(filter(str.isdigit, parts[1])))
        return gen, fak
    except: return 85, 15

# --- SESSION ---
if 'reviews_list' not in st.session_state: st.session_state.reviews_list = []
if 'meta' not in st.session_state: st.session_state.meta = ["-", "-", "-"]
if 'auth' not in st.session_state: st.session_state.auth = (0, 0)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=150)
    target_url = st.text_input("🔗 Amazon URL:")
    if st.button("🚀 UNLEASH AI", use_container_width=True):
        if target_url:
            with st.spinner("Analyzing Trends..."):
                revs, title, err = scrape_amazon(target_url)
                if revs:
                    st.session_state.reviews_list = revs
                    # AI Metadata Call
                    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                    meta_res = client.models.generate_content(model="gemini-2.0-flash", contents=f"Extract: Company | Model | Category. Title: {title}")
                    st.session_state.meta = meta_res.text.split('|')
                    st.session_state.auth = get_authenticity_score([r['text'] for r in revs])
                else: st.error(err)

st.markdown('<h1 class="gradient-text">SENTIMENT ANALYSIS</h1>', unsafe_allow_html=True)

if st.session_state.reviews_list:
    rev_data = st.session_state.reviews_list
    sia = SentimentIntensityAnalyzer()
    df = pd.DataFrame([{
        "Date": r['date'], 
        "Review": r['text'], 
        "Score": sia.polarity_scores(r['text'])['compound']
    } for r in rev_data])
    df = df.sort_values('Date')

    # Recommendation
    avg_s = df['Score'].mean()
    rec, col = ("MUST BUY", "#00ff88") if avg_s > 0.4 else (("GOOD BUY", "#FF9900") if avg_s > 0.05 else ("THINK AGAIN", "#ff3333"))

    # Panes
    m1, m2, m3, m4 = st.columns(4)
    meta = st.session_state.meta
    m1.markdown(f'<div class="metric-card"><p style="font-size:0.7rem; opacity:0.6;">COMPANY</p><h3 style="color:#FF9900">{meta[0] if len(meta)>0 else "N/A"}</h3></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><p style="font-size:0.7rem; opacity:0.6;">MODEL</p><h3 style="color:#FF9900">{meta[1] if len(meta)>1 else "N/A"}</h3></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><p style="font-size:0.7rem; opacity:0.6;">CATEGORY</p><h3 style="color:#FF9900">{meta[2] if len(meta)>2 else "N/A"}</h3></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><p style="font-size:0.7rem; opacity:0.6;">RECOMMENDATION</p><h2 style="color:{col};">{rec}</h2></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualization Row 1: DNA & Authenticity
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<p style="font-family:Orbitron; color:#FF9900;">🧬 PRODUCT DNA</p>', unsafe_allow_html=True)
        # Reuse previous Radar logic (simplified for space)
        dims = {'Quality':['quality','build'],'Value':['price','money'],'Usability':['easy','use'],'Durability':['last','strong'],'Service':['ship','pack']}
        v_scores = []
        for d, k in dims.items():
            rel = [r['text'] for r in rev_data if any(x in r['text'].lower() for x in k)]
            sc = (sum([sia.polarity_scores(r)['compound'] for r in rel])/len(rel)+1)/2 if rel else 0.5
            v_scores.append(round(sc*10, 1))
        
        fig_r = go.Figure(data=go.Scatterpolar(r=v_scores, theta=list(dims.keys()), fill='toself', marker=dict(color='#FF9900')))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 10]), bgcolor='rgba(0,0,0,0)'),
                            paper_bgcolor='rgba(0,0,0,0)', font_color="white", height=250, margin=dict(t=20,b=20,l=20,r=20))
        st.plotly_chart(fig_r, use_container_width=True)

    with col2:
        st.markdown('<p style="font-family:Orbitron; color:#FF9900;">⚖️ AUTHENTICITY SHARE</p>', unsafe_allow_html=True)
        gen, fak = st.session_state.auth
        fig_a = px.pie(values=[gen, fak], names=['Genuine', 'Fake'], hole=0.7, color_discrete_sequence=['#00ff88', '#ff3333'])
        fig_a.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", showlegend=False, height=250, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_a, use_container_width=True)

    # Visualization Row 2: Trend Analysis
    st.markdown('<p style="font-family:Orbitron; color:#FF9900;">📈 SENTIMENT TREND (MOMENTUM)</p>', unsafe_allow_html=True)
    df_trend = df.resample('ME', on='Date').mean().reset_index()
    fig_t = px.line(df_trend, x='Date', y='Score', markers=True)
    fig_t.update_traces(line_color='#FF9900', line_width=3, marker=dict(size=10, color="white"))
    fig_t.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", 
                        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, title="Sentiment Score"), height=300)
    st.plotly_chart(fig_t, use_container_width=True)

    # AI Neural Analyst (Bottom)
    st.markdown('<h3 style="color:#FF9900; font-family:Orbitron;">💬 NEURAL ANALYST</h3>', unsafe_allow_html=True)
    user_q = st.text_input("Ask about specific trends or features:")
    if user_q:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        ans = client.models.generate_content(model="gemini-2.0-flash", contents=f"Context: {str(rev_data)[:8000]}. Query: {user_q}")
        st.markdown(f'<div class="chat-box">{ans.text}</div>', unsafe_allow_html=True)

else:
    st.info("👋 System Standby. Awaiting URL.")
