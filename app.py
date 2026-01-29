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

# --- UI CONFIG & CLEAN STYLING ---
st.set_page_config(page_title="SENTIMENT ANALYSIS", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;600&display=swap');
    
    /* Pure White Background */
    .stApp {
        background-color: #ffffff;
        color: #1a1a1a;
        font-family: 'Inter', sans-serif;
    }
    
    /* Yellow Heading with Black Outline for visibility on white */
    .gradient-text {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.8rem;
        color: #FFCC00;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 20px;
        text-shadow: 
            -1px -1px 0 #000,  
             1px -1px 0 #000,
            -1px  1px 0 #000,
             1px  1px 0 #000;
    }

    /* Clean Card System with Shadows and Borders */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover {
        border-color: #FFCC00;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        transform: translateY(-5px);
    }

    /* DNA Scorecard Styling */
    .dna-container {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }

    .dna-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
    }
    .dna-table td {
        padding: 12px;
        border-bottom: 1px solid #f0f0f0;
    }
    .dna-label { font-weight: 600; color: #333333; }
    .dna-value { color: #d4a017; font-weight: bold; text-align: right; font-family: 'Orbitron'; }

    /* Clean Chat Box */
    .chat-box {
        background: #fdfdfd;
        border: 1px solid #e0e0e0;
        border-left: 5px solid #FFCC00;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        color: #1a1a1a;
    }

    /* Remove standard streamlit black borders on dataframes */
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- BACKEND LOGIC ---

def get_product_metadata(reviews, title):
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        prompt = f"Extract only the following 3 fields: 1. Company, 2. Model Name, 3. Category. Return as: Company | Model | Category. Context: {title} {str(reviews)[:2000]}"
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        parts = response.text.split('|')
        return [p.strip() for p in parts] if len(parts) == 3 else ["Unknown", "Unknown", "Unknown"]
    except:
        return ["N/A", "N/A", "N/A"]

def get_radar_data(reviews):
    dimensions = {
        'Quality': ['quality', 'build', 'premium', 'cheap', 'material'],
        'Value': ['price', 'worth', 'expensive', 'money', 'value'],
        'Usability': ['easy', 'use', 'setup', 'friendly'],
        'Durability': ['last', 'broke', 'sturdy', 'strong'],
        'Service': ['shipping', 'package', 'customer', 'delivery']
    }
    sia = SentimentIntensityAnalyzer()
    scores = []
    for dim, keywords in dimensions.items():
        relevant_revs = [r for r in reviews if any(k in r.lower() for k in keywords)]
        if not relevant_revs:
            scores.append(5.0)
        else:
            avg_score = sum([sia.polarity_scores(r)['compound'] for r in relevant_revs]) / len(relevant_revs)
            scores.append(round(((avg_score + 1) / 2) * 10, 1))
    return list(dimensions.keys()), scores

def get_ai_response(query, context):
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=f"Product Analyst. Context: {str(context)[:8000]}. Query: {query}"
        )
        return response.text
    except Exception as e: return f"AI Error: {str(e)}"

def scrape_amazon(url):
    try:
        api_key = st.secrets["SCRAPER_API_KEY"]
        payload = {'api_key': api_key, 'url': url}
        response = requests.get('http://api.scraperapi.com', params=payload, timeout=60)
        soup = BeautifulSoup(response.text, "html.parser")
        reviews = [el.get_text().strip() for el in soup.select('span[data-hook="review-body"]')]
        title = soup.find("span", {"id": "productTitle"})
        title_text = title.get_text().strip() if title else "Product"
        return reviews, title_text, None
    except Exception as e:
        return None, None, str(e)

# --- SESSION ---
if 'reviews_list' not in st.session_state: st.session_state.reviews_list = []
if 'meta' not in st.session_state: st.session_state.meta = ["-", "-", "-"]

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=120)
    st.markdown("<br>", unsafe_allow_html=True)
    target_url = st.text_input("🔗 Paste Amazon Review URL:")
    
    if st.button("🚀 UNLEASH AI", use_container_width=True):
        if target_url:
            with st.spinner("Processing Data..."):
                reviews, title, error = scrape_amazon(target_url)
                if reviews:
                    st.session_state.reviews_list = reviews
                    st.session_state.meta = get_product_metadata(reviews, title)
                else: st.error(error)

# --- DASHBOARD MAIN ---
st.markdown('<h1 class="gradient-text">SENTIMENT ANALYSIS</h1>', unsafe_allow_html=True)

if st.session_state.reviews_list:
    reviews = st.session_state.reviews_list
    sia = SentimentIntensityAnalyzer()
    df = pd.DataFrame([{"Review": r, "Score": sia.polarity_scores(r)['compound']} for r in reviews])
    df['Sentiment'] = df['Score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))
    
    avg_score = df['Score'].mean()
    rec_text, rec_color = ("MUST BUY", "#28a745") if avg_score > 0.4 else (("GOOD BUY", "#FF9900") if avg_score > 0.05 else ("THINK AGAIN", "#dc3545"))

    # --- PANES ---
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-card"><p style="font-size:0.8rem; opacity:0.6; color:#1a1a1a;">COMPANY</p><h3 style="color:#d4a017">{st.session_state.meta[0]}</h3></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><p style="font-size:0.8rem; opacity:0.6; color:#1a1a1a;">MODEL</p><h3 style="color:#d4a017">{st.session_state.meta[1]}</h3></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><p style="font-size:0.8rem; opacity:0.6; color:#1a1a1a;">CATEGORY</p><h3 style="color:#d4a017">{st.session_state.meta[2]}</h3></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><p style="font-size:0.8rem; opacity:0.6; color:#1a1a1a;">RECOMMENDATION</p><h2 style="color:{rec_color}; font-weight:bold;">{rec_text}</h2></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row with Shadows/Borders
    col_radar, col_dna_list = st.columns([1.5, 1])
    labels, values = get_radar_data(reviews)

    with col_radar:
        st.markdown('<div class="dna-container">', unsafe_allow_html=True)
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values, theta=labels, fill='toself',
            marker=dict(color='#FFCC00'), line=dict(color='#FFCC00')
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10], gridcolor="#f0f0f0"), bgcolor='rgba(255,255,255,0)'),
            paper_bgcolor='rgba(255,255,255,0)', font_color="#1a1a1a", title="Product DNA Visualization"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_dna_list:
        st.markdown('<div class="dna-container">', unsafe_allow_html=True)
        st.markdown('<p style="font-family:Orbitron; color:#d4a017; font-weight:bold;">🧬 DNA SCORECARD</p>', unsafe_allow_html=True)
        dna_html = '<table class="dna-table">'
        for label, val in zip(labels, values):
            dna_html += f'<tr><td class="dna-label">{label}</td><td class="dna-value">{val}/10</td></tr>'
        dna_html += '</table>'
        st.markdown(dna_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Visual Sentiment Share
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="dna-container">', unsafe_allow_html=True)
    fig_pie = px.pie(df, names='Sentiment', hole=0.7, 
                     color='Sentiment', color_discrete_map={'Positive':'#28a745','Negative':'#dc3545','Neutral':'#e0e0e0'})
    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="#1a1a1a", showlegend=True, height=350, title="Sentiment Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Neural Analyst
    st.markdown('<br><h3 style="color:#d4a017; font-family:Orbitron;">💬 NEURAL ANALYST</h3>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    if c1.button("✅ Quick Pros", use_container_width=True):
        st.session_state.chat_answer = get_ai_response("Top 3 pros?", reviews)
    if c2.button("❌ Quick Cons", use_container_width=True):
        st.session_state.chat_answer = get_ai_response("Top 3 cons?", reviews)

    user_query = st.text_input("Interrogate the data:")
    if user_query:
        with st.spinner("Processing..."):
            st.session_state.chat_answer = get_ai_response(user_query, reviews)
            
    if st.session_state.get('chat_answer'):
        st.markdown(f'<div class="chat-box">{st.session_state.chat_answer}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df.style.background_gradient(cmap='YlGn', subset=['Score']), use_container_width=True)

else:
    st.info("👋 System Standby. Awaiting URL Input.")
