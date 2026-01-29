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

# --- UI CONFIG & LIGHT THEME ---
st.set_page_config(page_title="SENTIMENT ANALYSIS", layout="wide")

# This CSS forces the app to stay light and removes all black containers
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Inter:wght@400;600&display=swap');
    
    /* Force Light Background on Everything */
    .stApp, [data-testid="stSidebar"], .main {
        background-color: #ffffff !important;
        color: #2D2D2D !important;
        font-family: 'Inter', sans-serif;
    }

    /* Black-outlined Yellow Header */
    .gradient-text {
        font-family: 'Orbitron', sans-serif;
        font-size: 3rem;
        color: #FFCC00;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 30px;
        text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;
    }

    /* Clean Card System */
    .metric-card {
        background: #ffffff;
        border: 1px solid #EAEAEA;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        color: #2D2D2D !important;
    }
    
    .metric-card h3, .metric-card h2, .metric-card p {
        color: #2D2D2D !important;
        margin: 5px 0;
    }

    /* DNA Scorecard */
    .dna-container {
        background: #ffffff;
        border: 1px solid #EAEAEA;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    .dna-table {
        width: 100%;
        border-collapse: collapse;
    }
    .dna-table td {
        padding: 12px;
        border-bottom: 1px solid #F5F5F5;
        color: #444 !important;
    }
    .dna-label { font-weight: 600; }
    .dna-value { color: #d4a017; font-weight: bold; font-family: 'Orbitron'; text-align: right; }

    /* Neural Analyst Chat Box */
    .chat-box {
        background: #F9F9FB;
        border: 1px solid #EAEAEA;
        border-left: 6px solid #FFCC00;
        padding: 20px;
        border-radius: 8px;
        color: #2D2D2D !important;
        margin-top: 15px;
    }

    /* Sidebar Fixes */
    [data-testid="stSidebar"] .stMarkdown p { color: #2D2D2D !important; }
    
    /* Table Fixes */
    .stDataFrame { background: white; border: 1px solid #EAEAEA; border-radius: 12px; }
    
    /* Heading Colors */
    h1, h2, h3 { color: #2D2D2D !important; }
    </style>
    """, unsafe_allow_html=True)

# --- BACKEND LOGIC ---
def get_product_metadata(reviews, title):
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        prompt = f"Extract: Company | Model | Category. Context: {title} {str(reviews)[:2000]}"
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        parts = response.text.split('|')
        return [p.strip() for p in parts] if len(parts) == 3 else ["Unknown", "Unknown", "Unknown"]
    except: return ["N/A", "N/A", "N/A"]

def get_radar_data(reviews):
    dimensions = {'Quality':['quality','build'],'Value':['price','worth'],'Usability':['easy','use'],'Durability':['last','sturdy'],'Service':['ship','customer']}
    sia = SentimentIntensityAnalyzer()
    scores = []
    for dim, keywords in dimensions.items():
        rel = [r for r in reviews if any(k in r.lower() for k in keywords)]
        s = round(((sum([sia.polarity_scores(r)['compound'] for r in rel])/len(rel)+1)/2)*10,1) if rel else 5.0
        scores.append(s)
    return list(dimensions.keys()), scores

def scrape_amazon(url):
    try:
        api_key = st.secrets["SCRAPER_API_KEY"]
        res = requests.get('http://api.scraperapi.com', params={'api_key': api_key, 'url': url}, timeout=60)
        soup = BeautifulSoup(res.text, "html.parser")
        revs = [el.get_text().strip() for el in soup.select('span[data-hook="review-body"]')]
        title = soup.find("span", {"id": "productTitle"})
        return revs, (title.get_text().strip() if title else "Product"), None
    except Exception as e: return None, None, str(e)

# --- APP START ---
if 'reviews_list' not in st.session_state: st.session_state.reviews_list = []
if 'meta' not in st.session_state: st.session_state.meta = ["-", "-", "-"]

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=120)
    st.write("---")
    url = st.text_input("🔗 Amazon URL:")
    if st.button("RUN ANALYSIS", use_container_width=True):
        if url:
            with st.spinner("Analyzing..."):
                revs, title, err = scrape_amazon(url)
                if revs:
                    st.session_state.reviews_list = revs
                    st.session_state.meta = get_product_metadata(revs, title)
                else: st.error(err)

st.markdown('<h1 class="gradient-text">SENTIMENT ANALYSIS</h1>', unsafe_allow_html=True)

if st.session_state.reviews_list:
    reviews = st.session_state.reviews_list
    sia = SentimentIntensityAnalyzer()
    df = pd.DataFrame([{"Review": r, "Score": sia.polarity_scores(r)['compound']} for r in reviews])
    
    # 4-Column Layout
    m1, m2, m3, m4 = st.columns(4)
    meta = st.session_state.meta
    m1.markdown(f'<div class="metric-card"><p>COMPANY</p><h3>{meta[0]}</h3></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><p>MODEL</p><h3>{meta[1]}</h3></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><p>CATEGORY</p><h3>{meta[2]}</h3></div>', unsafe_allow_html=True)
    
    avg = df['Score'].mean()
    rec, color = ("MUST BUY", "#28a745") if avg > 0.4 else (("GOOD BUY", "#FF9900") if avg > 0.05 else ("THINK AGAIN", "#dc3545"))
    m4.markdown(f'<div class="metric-card"><p>DECISION</p><h2 style="color:{color} !important;">{rec}</h2></div>', unsafe_allow_html=True)

    st.write("<br>", unsafe_allow_html=True)

    # Visualization Row
    col_radar, col_breakdown = st.columns([1.5, 1])
    labels, values = get_radar_data(reviews)

    with col_radar:
        st.markdown('<div class="dna-container">', unsafe_allow_html=True)
        st.subheader("🧬 Product DNA")
        fig = go.Figure(data=go.Scatterpolar(r=values, theta=labels, fill='toself', fillcolor='rgba(255, 204, 0, 0.2)', line=dict(color='#FFCC00')))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10], gridcolor="#eee"), bgcolor='rgba(255,255,255,0)'),
                          paper_bgcolor='rgba(255,255,255,0)', font_color="#2D2D2D", height=380, margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_breakdown:
        st.markdown('<div class="dna-container">', unsafe_allow_html=True)
        st.subheader("📊 Breakdown")
        dna_html = '<table class="dna-table">'
        for l, v in zip(labels, values):
            dna_html += f'<tr><td class="dna-label">{l}</td><td class="dna-value">{v}/10</td></tr>'
        dna_html += '</table>'
        st.markdown(dna_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Pie Chart
        fig_pie = px.pie(df, names=df['Score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')), 
                         hole=0.6, color_discrete_map={'Positive':'#28a745','Negative':'#dc3545','Neutral':'#EAEAEA'})
        fig_pie.update_layout(height=200, showlegend=False, margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)

    # Neural Analyst Section
    st.markdown('<div class="dna-container">', unsafe_allow_html=True)
    st.subheader("💬 Neural Analyst")
    user_q = st.text_input("Ask a question about these reviews:")
    if user_q:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        ans = client.models.generate_content(model="gemini-2.0-flash", contents=f"Data: {str(reviews)[:5000]}. Q: {user_q}")
        st.markdown(f'<div class="chat-box">{ans.text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.dataframe(df, use_container_width=True)
else:
    st.info("👋 System Ready. Please provide an Amazon Review URL in the sidebar.")
