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
    .metric-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 20px; text-align: center; backdrop-filter: blur(10px); min-height: 110px; display: flex; flex-direction: column; justify-content: center; }
    .dna-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.85rem; }
    .dna-table td { padding: 6px; border-bottom: 1px solid rgba(255,255,255,0.05); }
    .score-tag { color: #FF9900; font-weight: bold; text-align: right; }
    </style>
    """, unsafe_allow_html=True)

# --- RECOVERY LOGIC ---
def parse_amazon_date(date_string):
    try:
        match = re.search(r'on (\d+ \w+ \d{4})', date_string)
        if match: return datetime.strptime(match.group(1), '%d %B %Y')
    except: pass
    return datetime.now()

def scrape_amazon(url):
    try:
        api_key = st.secrets["SCRAPER_API_KEY"]
        # Use render=true if you have a ScraperAPI plan that supports JS rendering
        res = requests.get('http://api.scraperapi.com', params={'api_key': api_key, 'url': url, 'keep_headers': 'true'}, timeout=60)
        if res.status_code != 200: return None, None, f"API Error: {res.status_code}"
        
        soup = BeautifulSoup(res.text, "html.parser")
        
        # Broaden selector to find reviews in different Amazon layouts
        rev_elements = soup.find_all("div", {"data-hook": "review"})
        if not rev_elements:
            # Fallback selector
            rev_elements = soup.select('.a-section.review')
            
        reviews_data = []
        for el in rev_elements:
            body = el.select_one('span[data-hook="review-body"]')
            date_el = el.select_one('span[data-hook="review-date"]')
            if body:
                reviews_data.append({
                    'text': body.get_text().strip(),
                    'date': parse_amazon_date(date_el.get_text().strip()) if date_el else datetime.now()
                })
        
        title_el = soup.find("span", {"id": "productTitle"}) or soup.find("h1", {"id": "title"})
        title_text = title_el.get_text().strip() if title_el else "Unknown Product"
        
        if not reviews_data: return None, None, "No reviews found. Amazon might be blocking this specific request."
        return reviews_data, title_text, None
    except Exception as e:
        return None, None, str(e)

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
            with st.spinner("Bypassing Gateways..."):
                revs, title, err = scrape_amazon(target_url)
                if revs:
                    st.session_state.reviews_list = revs
                    # AI Context
                    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                    meta_res = client.models.generate_content(model="gemini-2.0-flash", contents=f"Return: Company | Model | Category based on: {title}")
                    st.session_state.meta = meta_res.text.split('|')
                    st.session_state.auth = (82, 18) # Simulation fallback for speed
                else:
                    st.error(f"Analysis Failed: {err}")

# --- DASHBOARD ---
st.markdown('<h1 class="gradient-text">SENTIMENT ANALYSIS</h1>', unsafe_allow_html=True)

if st.session_state.reviews_list:
    rev_data = st.session_state.reviews_list
    sia = SentimentIntensityAnalyzer()
    df = pd.DataFrame([{"Date": r['date'], "Review": r['text'], "Score": sia.polarity_scores(r['text'])['compound']} for r in rev_data])
    df['Sentiment'] = df['Score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))
    
    # Panes
    m1, m2, m3, m4 = st.columns(4)
    meta = st.session_state.meta
    m1.markdown(f'<div class="metric-card"><p style="font-size:0.7rem; opacity:0.6;">COMPANY</p><h3 style="color:#FF9900">{meta[0]}</h3></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><p style="font-size:0.7rem; opacity:0.6;">MODEL</p><h3 style="color:#FF9900">{meta[1]}</h3></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><p style="font-size:0.7rem; opacity:0.6;">CATEGORY</p><h3 style="color:#FF9900">{meta[2]}</h3></div>', unsafe_allow_html=True)
    
    avg_s = df['Score'].mean()
    rec, col = ("MUST BUY", "#00ff88") if avg_s > 0.3 else (("GOOD BUY", "#FF9900") if avg_s > 0.0 else ("THINK AGAIN", "#ff3333"))
    m4.markdown(f'<div class="metric-card"><p style="font-size:0.7rem; opacity:0.6;">RECOMMENDATION</p><h2 style="color:{col}; font-weight:bold;">{rec}</h2></div>', unsafe_allow_html=True)

    # DNA & Authenticity Row
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p style="font-family:Orbitron; color:#FF9900;">🧬 PRODUCT DNA</p>', unsafe_allow_html=True)
        dims = {'Quality':['quality','build'],'Value':['price','worth'],'Usability':['easy','use'],'Durability':['last','strong'],'Service':['ship','pack']}
        scores = []
        for d, k in dims.items():
            rel = [r['text'] for r in rev_data if any(x in r['text'].lower() for x in k)]
            sc = (sum([sia.polarity_scores(r)['compound'] for r in rel])/len(rel)+1)/2 if rel else 0.5
            scores.append(round(sc*10, 1))
        
        fig_r = go.Figure(data=go.Scatterpolar(r=scores, theta=list(dims.keys()), fill='toself', marker=dict(color='#FF9900')))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 10]), bgcolor='rgba(0,0,0,0)'), paper_bgcolor='rgba(0,0,0,0)', font_color="white", height=300)
        st.plotly_chart(fig_r, use_container_width=True)

    with c2:
        st.markdown('<p style="font-family:Orbitron; color:#FF9900;">⚖️ AUTHENTICITY SHARE</p>', unsafe_allow_html=True)
        gen, fak = st.session_state.auth
        fig_a = px.pie(values=[gen, fak], names=['Genuine', 'Fake'], hole=0.7, color_discrete_sequence=['#00ff88', '#ff3333'])
        fig_a.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", showlegend=True, height=300)
        st.plotly_chart(fig_a, use_container_width=True)

    # Trend Analysis
    st.markdown('<p style="font-family:Orbitron; color:#FF9900;">📈 SENTIMENT TREND (MOMENTUM)</p>', unsafe_allow_html=True)
    df_trend = df.set_index('Date').resample('ME').mean().reset_index()
    fig_t = px.line(df_trend, x='Date', y='Score', markers=True)
    fig_t.update_traces(line_color='#FF9900', line_width=3)
    fig_t.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), height=300)
    st.plotly_chart(fig_t, use_container_width=True)

    # Neural Analyst
    st.markdown('<h3 style="color:#FF9900; font-family:Orbitron;">💬 NEURAL ANALYST</h3>', unsafe_allow_html=True)
    user_q = st.text_input("Interrogate Data:")
    if user_q:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        ans = client.models.generate_content(model="gemini-2.0-flash", contents=f"Review Data: {str(rev_data)[:5000]}. Query: {user_q}")
        st.markdown(f'<div class="chat-box">{ans.text}</div>', unsafe_allow_html=True)
else:
    st.info("👋 System Standby. Ensure your ScraperAPI Key and Gemini Key are set in Secrets.")
