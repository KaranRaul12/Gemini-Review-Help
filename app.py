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

# --- UI CONFIG & CLEAN THEME ---
st.set_page_config(page_title="SENTIMENT ANALYSIS", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Inter:wght@400;600&display=swap');
    
    /* Light Grey/White Background */
    .stApp {
        background-color: #f8f9fa;
        color: #1a1a1a;
        font-family: 'Inter', sans-serif;
    }
    
    /* Yellow Heading with Black Outline */
    .gradient-text {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.2rem;
        color: #FFCC00;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 25px;
        text-shadow: 
            -2px -2px 0 #000,  
             2px -2px 0 #000,
            -2px  2px 0 #000,
             2px  2px 0 #000;
    }

    /* Clean Card System */
    .metric-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #FFCC00;
    }

    /* Section Containers */
    .section-container {
        background: white;
        border: 1px solid #ddd;
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }

    .dna-table {
        width: 100%;
        border-collapse: collapse;
    }
    .dna-table td {
        padding: 12px;
        border-bottom: 1px solid #eee;
        color: #444;
    }
    .dna-label { font-weight: 600; }
    .dna-value { color: #d4a017; font-weight: bold; text-align: right; font-family: 'Orbitron'; }

    .chat-box {
        background: #fff9db;
        border: 1px solid #ffe066;
        border-left: 6px solid #FFCC00;
        padding: 20px;
        border-radius: 10px;
        color: #5c4d00;
    }
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
    st.markdown("### Control Panel")
    url = st.text_input("🔗 Product Link:")
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
    
    # Header Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-card"><small>COMPANY</small><br><b>{st.session_state.meta[0]}</b></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><small>MODEL</small><br><b>{st.session_state.meta[1]}</b></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><small>CATEGORY</small><br><b>{st.session_state.meta[2]}</b></div>', unsafe_allow_html=True)
    
    avg = df['Score'].mean()
    rec, color = ("MUST BUY", "#28a745") if avg > 0.4 else (("GOOD BUY", "#FF9900") if avg > 0.05 else ("THINK AGAIN", "#dc3545"))
    m4.markdown(f'<div class="metric-card"><small>DECISION</small><br><b style="color:{color}">{rec}</b></div>', unsafe_allow_html=True)

    st.write("---")

    # Body Sections
    col_left, col_right = st.columns([1.5, 1])
    
    with col_left:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.subheader("🧬 Product DNA")
        labels, values = get_radar_data(reviews)
        fig = go.Figure(data=go.Scatterpolar(r=values, theta=labels, fill='toself', fillcolor='rgba(255, 204, 0, 0.3)', line=dict(color='#FFCC00')))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.subheader("📊 Breakdown")
        dna_html = '<table class="dna-table">'
        for l, v in zip(labels, values):
            dna_html += f'<tr><td class="dna-label">{l}</td><td class="dna-value">{v}/10</td></tr>'
        dna_html += '</table>'
        st.markdown(dna_html, unsafe_allow_html=True)
        
        st.write("---")
        fig_p = px.pie(df, names=df['Score'].apply(lambda x: 'Pos' if x > 0.05 else ('Neg' if x < -0.05 else 'Neu')), hole=0.5, color_discrete_sequence=['#28a745','#dc3545','#999'])
        fig_p.update_layout(height=250, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_p, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("💬 Neural Analyst")
    q = st.text_input("Ask anything about this product:")
    if q:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        ans = client.models.generate_content(model="gemini-2.0-flash", contents=f"Context: {str(reviews)[:6000]}. Query: {q}")
        st.markdown(f'<div class="chat-box">{ans.text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.dataframe(df, use_container_width=True)
else:
    st.info("👋 Dashboard Ready. Paste a URL in the sidebar to begin analysis.")
