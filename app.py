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

# --- UI CONFIG & ADVANCED STYLING ---
st.set_page_config(page_title="Amazon AI Insights Pro", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;600&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 50% 50%, #12141d 0%, #050505 100%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Neon Header */
    .gradient-text {
        background: linear-gradient(92deg, #FF9900 0%, #FF5F6D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        letter-spacing: 2px;
    }

    /* Glassmorphism Tiles */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .metric-card:hover {
        border-color: #FF9900;
        box-shadow: 0 0 20px rgba(255, 153, 0, 0.2);
        transform: translateY(-8px);
    }

    /* Modern Chat UI */
    .chat-box {
        background: linear-gradient(145deg, rgba(28,31,43,1) 0%, rgba(14,17,23,1) 100%);
        border: 1px solid #333;
        border-left: 4px solid #FF9900;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 10px 10px 30px rgba(0,0,0,0.5);
        line-height: 1.6;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #FF9900; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- BACKEND LOGIC ---
def get_radar_data(reviews):
    """Calculates specific dimension scores using keyword mapping."""
    dimensions = {
        'Quality': ['quality', 'build', 'premium', 'cheap', 'material'],
        'Value': ['price', 'worth', 'expensive', 'cheap', 'money', 'value'],
        'Usability': ['easy', 'use', 'setup', 'complicated', 'friendly'],
        'Durability': ['last', 'broke', 'sturdy', 'strong', 'long-term'],
        'Service': ['shipping', 'package', 'customer', 'arrived', 'delivery']
    }
    sia = SentimentIntensityAnalyzer()
    scores = []
    for dim, keywords in dimensions.items():
        relevant_revs = [r for r in reviews if any(k in r.lower() for k in keywords)]
        if not relevant_revs:
            scores.append(0.5) # Neutral base
        else:
            avg_score = sum([sia.polarity_scores(r)['compound'] for r in relevant_revs]) / len(relevant_revs)
            # Normalize to 0-1 scale
            scores.append((avg_score + 1) / 2)
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

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=150)
    st.markdown("<br>", unsafe_allow_html=True)
    target_url = st.text_input("🔗 Paste Amazon Review URL:")
    analyze_btn = st.button("🚀 UNLEASH AI", use_container_width=True)
    
    if analyze_btn and target_url:
        with st.spinner("Processing..."):
            # Reuse previous scrape logic
            api_key = st.secrets["SCRAPER_API_KEY"]
            res = requests.get('http://api.scraperapi.com', params={'api_key': api_key, 'url': target_url})
            soup = BeautifulSoup(res.text, "html.parser")
            st.session_state.reviews_list = [el.get_text().strip() for el in soup.select('span[data-hook="review-body"]')]
            st.session_state.chat_answer = ""

# --- DASHBOARD MAIN ---
st.markdown('<h1 class="gradient-text">INSIGHT ENGINE PRO</h1>', unsafe_allow_html=True)

if st.session_state.get('reviews_list'):
    reviews = st.session_state.reviews_list
    sia = SentimentIntensityAnalyzer()
    df = pd.DataFrame([{"Review": r, "Score": sia.polarity_scores(r)['compound']} for r in reviews])
    df['Sentiment'] = df['Score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-card"><p>Total Feed</p><h2>{len(df)}</h2></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><p>Positive</p><h2 style="color:#00ff88">{len(df[df.Sentiment=="Positive"])}</h2></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><p>Negative</p><h2 style="color:#ff3333">{len(df[df.Sentiment=="Negative"])}</h2></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><p>Avg Score</p><h2 style="color:#FF9900">{df.Score.mean():.2f}</h2></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    col_radar, col_pie = st.columns([1.2, 1])
    
    with col_radar:
        labels, values = get_radar_data(reviews)
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values, theta=labels, fill='toself',
            marker=dict(color='#FF9900'), line=dict(color='#FF9900')
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[0, 1]), bgcolor='rgba(0,0,0,0)'),
            paper_bgcolor='rgba(0,0,0,0)', font_color="white", title="Product Feature DNA"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_pie:
        fig_pie = px.pie(df, names='Sentiment', hole=0.7, 
                         color='Sentiment', color_discrete_map={'Positive':'#00ff88','Negative':'#ff3333','Neutral':'#444'})
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<h3 style="color:#FF9900; font-family:Orbitron;">💬 NEURAL ANALYST</h3>', unsafe_allow_html=True)
    
    # AI Interaction
    user_query = st.text_input("Interrogate the data:", placeholder="E.g. Identify any recurring build-quality issues...")
    if user_query:
        with st.spinner("Processing Neural Pathways..."):
            st.session_state.chat_answer = get_ai_response(user_query, reviews)
            
    if st.session_state.chat_answer:
        st.markdown(f'<div class="chat-box">{st.session_state.chat_answer}</div>', unsafe_allow_html=True)

    # Data Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df.style.background_gradient(cmap='RdYlGn', subset=['Score']), use_container_width=True)

else:
    st.info("👋 System Standby. Awaiting URL Input in Control Panel.")
