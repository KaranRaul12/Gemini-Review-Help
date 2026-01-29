import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from google import genai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# ---------------- NLTK ----------------
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Product Sentiment Intelligence", layout="wide")

# ---------------- UI CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

:root {
    --bg: #0b0f14;
    --card: rgba(255,255,255,0.06);
    --border: rgba(255,255,255,0.12);
    --accent: #ff9900;
    --muted: rgba(255,255,255,0.65);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #0b0f14, #050608);
    color: white;
}

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 22px;
    transition: 0.25s;
}

.card:hover {
    transform: translateY(-4px);
    border-color: var(--accent);
}

.metric-label {
    font-size: 0.75rem;
    letter-spacing: 1px;
    color: var(--muted);
}

.metric-value {
    font-size: 1.3rem;
    font-weight: 600;
    margin-top: 6px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1218, #07090d);
}
</style>
""", unsafe_allow_html=True)

# ---------------- UI HELPER ----------------
def metric_card(label, value, color="#ffffff"):
    st.markdown(f"""
    <div class="card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- BACKEND ----------------
def scrape_amazon(url):
    try:
        payload = {
            "api_key": st.secrets["SCRAPER_API_KEY"],
            "url": url
        }
        r = requests.get("http://api.scraperapi.com", params=payload, timeout=60)
        soup = BeautifulSoup(r.text, "html.parser")
        reviews = [x.get_text().strip() for x in soup.select('span[data-hook="review-body"]')]
        title = soup.find("span", {"id": "productTitle"})
        return reviews, title.get_text().strip() if title else "Product", None
    except Exception as e:
        return None, None, str(e)

def get_product_metadata(reviews, title):
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        prompt = f"Extract Company | Model | Category from product: {title}"
        res = client.models.generate_content("gemini-2.0-flash", prompt)
        parts = res.text.split("|")
        return [p.strip() for p in parts] if len(parts) == 3 else ["Unknown"] * 3
    except:
        return ["N/A", "N/A", "N/A"]

def detect_fake_reviews_svm(reviews):
    sia = SentimentIntensityAnalyzer()

    labels = []
    for r in reviews:
        score = sia.polarity_scores(r)["compound"]
        if abs(score) > 0.85 or len(r.split()) < 6:
            labels.append(1)  # Fake
        else:
            labels.append(0)  # Genuine

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=3000)),
        ("svm", SVC(kernel="linear", probability=True))
    ])

    model.fit(reviews, labels)
    preds = model.predict(reviews)

    fake = sum(preds)
    genuine = len(preds) - fake
    return fake, genuine

# ---------------- SESSION ----------------
if "reviews" not in st.session_state:
    st.session_state.reviews = []
if "meta" not in st.session_state:
    st.session_state.meta = ["-", "-", "-"]

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🔗 Amazon Review URL")
    url = st.text_input("Paste product link")

    if st.button("Analyze"):
        if url:
            with st.spinner("Analyzing reviews..."):
                reviews, title, err = scrape_amazon(url)
                if reviews:
                    st.session_state.reviews = reviews
                    st.session_state.meta = get_product_metadata(reviews, title)
                else:
                    st.error(err)

# ---------------- MAIN ----------------
st.markdown("## AI Product Sentiment Intelligence")
st.caption("Sentiment analysis + Fake review detection using NLP & SVM")

if st.session_state.reviews:
    sia = SentimentIntensityAnalyzer()

    df = pd.DataFrame([
        {"Review": r, "Score": sia.polarity_scores(r)["compound"]}
        for r in st.session_state.reviews
    ])

    df["Sentiment"] = df["Score"].apply(
        lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral"
    )

    avg = df["Score"].mean()
    rec, rec_color = (
        ("MUST BUY", "#00ff88") if avg > 0.4 else
        ("GOOD BUY", "#ff9900") if avg > 0.05 else
        ("THINK AGAIN", "#ff3333")
    )

    fake_cnt, genuine_cnt = detect_fake_reviews_svm(st.session_state.reviews)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("COMPANY", st.session_state.meta[0], "#ff9900")
    with c2: metric_card("MODEL", st.session_state.meta[1], "#ff9900")
    with c3: metric_card("CATEGORY", st.session_state.meta[2], "#ff9900")
    with c4: metric_card("RECOMMENDATION", rec, rec_color)
    with c5: metric_card("FAKE REVIEWS", fake_cnt, "#ff3333")

    st.markdown("### 📊 Sentiment Distribution")
    st.plotly_chart(px.pie(df, names="Sentiment", hole=0.7), use_container_width=True)

    st.markdown("### 🚨 Fake vs Genuine Reviews")
    fake_df = pd.DataFrame({
        "Type": ["Fake", "Genuine"],
        "Count": [fake_cnt, genuine_cnt]
    })

    st.plotly_chart(
        px.pie(fake_df, names="Type", values="Count", hole=0.6,
               color="Type",
               color_discrete_map={"Fake": "#ff3333", "Genuine": "#00ff88"}),
        use_container_width=True
    )

    st.dataframe(df, use_container_width=True)

    st.caption(
        "⚠️ Fake reviews are detected using a weakly-supervised SVM model trained on "
        "sentiment extremeness and linguistic patterns due to lack of labeled ground truth."
    )
else:
    st.info("Awaiting Amazon product URL")
