#!/usr/bin/env python3

import json
import re
import numpy as np
import faiss
import imagehash
import streamlit as st

from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer
from groq import Groq
import base64, io
import time
import pickle
# =========================
# CONFIG
# =========================
import os

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

groq_client = Groq(api_key=GROQ_API_KEY)
# st.sidebar.write("🔑 Key loaded:", GROQ_API_KEY[:8] + "...")
# MEME_FOLDER = r"C:\Users\Muhammad Shiraz\OneDrive\Desktop\Hackathon\memes"
MEME_FOLDER = "memes"

embedder = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# LOAD IMAGES
# =========================

def load_images(folder):
    folder = Path(folder)
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
    images = []
    for e in exts:
        images += list(folder.rglob(e))
    return list(set(images))


# =========================
# DUPLICATE CHECK
# =========================

def is_duplicate(image_path, seen):
    try:
        img = Image.open(image_path)
        h = imagehash.average_hash(img)
        if h in seen:
            return True
        seen.add(h)
        return False
    except:
        return False


# =========================
# GROQ VISION ANALYSIS
# =========================

PROMPT = """
You are a meme analyzer.

Return ONLY valid JSON:

{
  "text_in_image": "",
  "visual_description": "",
  "category": "",
  "emotion": "",
  "keywords": [],
  "summary": "",
  "title": "",
  "funniness": 5
}

funniness must be an integer from 1 (not funny) to 10 (extremely funny).
title should be a short, descriptive name for this meme.
"""


def safe_json_parse(text):
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except:
        return {
            "text_in_image": "",
            "visual_description": text[:200],
            "category": "general",
            "emotion": "unknown",
            "keywords": [],
            "summary": text[:200],
            "title": "",
            "funniness": 5
        }

def analyze_with_gemini(image_path):
    time.sleep(2)  # 2 seconds between each call

    for attempt in range(3):  # retry up to 3 times
        try:
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            b64 = base64.b64encode(buffer.getvalue()).decode()

            response = groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }],
                max_tokens=500
            )
            return safe_json_parse(response.choices[0].message.content)

        except Exception as e:
            error_msg = str(e)
            print(f"Attempt {attempt+1} failed: {error_msg}")

            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                wait = (attempt + 1) * 10  # wait 10s, 20s, 30s
                print(f"Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                time.sleep(3)

    return {
        "text_in_image": "", "visual_description": "",
        "category": "general", "emotion": "unknown",
        "keywords": [], "summary": "failed",
        "title": "", "funniness": 5
    }


# =========================
# BUILD EMBEDDING TEXT
# =========================

def build_text(data):
    category = data.get("category", "") or ""
    emotion = data.get("emotion", "") or ""
    keywords = " ".join(data.get("keywords", []) or [])

    # Repeat category/emotion/keywords so they carry more weight in the
    # embedding than incidental description text. Plain string-concat
    # embeddings average everything equally, so without this a meme's
    # category can get diluted into irrelevance.
    return " ".join([
        (category + " ") * 3,
        (emotion + " ") * 2,
        (keywords + " ") * 2,
        data.get("text_in_image", ""),
        data.get("visual_description", ""),
        data.get("summary", ""),
        data.get("title", ""),
        str(data.get("funniness", ""))
    ])


# =========================
# BUILD INDEX  (cached so it only runs once per session)
# =========================

@st.cache_resource(show_spinner=False)
def build_index(folder):
    import pickle

    # Load from disk if exists
    if Path("meme_index.faiss").exists() and Path("meme_meta.pkl").exists():
        index = faiss.read_index("meme_index.faiss")
        with open("meme_meta.pkl", "rb") as f:
            metadata = pickle.load(f)
        return index, metadata

    images = load_images(folder)
    vectors = []
    metadata = []
    seen = set()

    progress = st.progress(0, text="Building meme index…")
    total = len(images)

    for i, img in enumerate(images):
        progress.progress((i + 1) / max(total, 1), text=f"Processing {img.name} ({i+1}/{total})")

        if is_duplicate(img, seen):
            continue

        data = analyze_with_gemini(img)
        text = build_text(data)
        vec = embedder.encode(text).astype("float32")
        vectors.append(vec)
        metadata.append({"path": str(img), "data": data})

    progress.empty()

    if not vectors:
        return None, None

    vectors = np.array(vectors).astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # Save to disk
    faiss.write_index(index, "meme_index.faiss")
    with open("meme_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)

    return index, metadata


# =========================
# SEARCH
# =========================

def search(query, index, metadata, k=6):
    q = embedder.encode(query).astype("float32").reshape(1, -1)
    faiss.normalize_L2(q)

    # --- Stage 1: get the top 5 raw vector matches ---------------------
    # These 5 are used purely to decide what "theme" the person is
    # actually asking for (e.g. category="Dark Humor"), since a single
    # closest match can be a fluke but 5 agreeing on a category is signal.
    top5_n = min(5, len(metadata))
    top5_scores, top5_idxs = index.search(q, top5_n)

    from collections import Counter
    cat_counter = Counter()
    for i, idx in enumerate(top5_idxs[0]):
        if idx == -1:
            continue
        d = metadata[idx].get("data", {}) or {}
        cat = (d.get("category") or "").strip().lower()
        if cat:
            cat_counter[cat] += 1

    dominant_category = cat_counter.most_common(1)[0][0] if cat_counter else None

    # --- Stage 2: pull a wider pool, score it, boost matches that share
    # the dominant category (and any literal keyword overlap with the
    # query), then return the top k -------------------------------------
    pool = min(len(metadata), max(k * 5, 25))
    scores, idxs = index.search(q, pool)

    query_words = set(re.findall(r"[a-z']+", query.lower()))

    results = []
    for i, idx in enumerate(idxs[0]):
        if idx == -1:
            continue
        item = dict(metadata[idx])
        base_score = float(scores[0][i])

        d = item.get("data", {}) or {}
        cat = (d.get("category") or "").strip().lower()
        tag_words = set(re.findall(
            r"[a-z']+",
            " ".join([cat, d.get("emotion", "") or "",
                       " ".join(d.get("keywords", []) or [])]).lower()
        ))
        overlap = len(query_words & tag_words)

        boosted_score = base_score + (0.05 * overlap)
        if dominant_category and cat == dominant_category:
            boosted_score += 0.12  # keep results on-theme with the top-5 consensus

        item["score"] = min(boosted_score, 1.0)
        item["_base_score"] = base_score
        results.append(item)

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Meme Finder · Semantic Meme Search", page_icon="◆", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #0e1118;
    --surface: #151a24;
    --surface-2: #1b212d;
    --line: #262d3a;
    --ink: #eef1f6;
    --muted: #7e8a9c;
    --coral: #ff6b4a;
    --mint: #57d9b8;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family: 'Inter', sans-serif !important;
    height: 100vh !important;
    overflow: hidden !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { right: 1rem; }
[data-testid="stAppViewContainer"] > .main {
    height: 100vh !important;
    overflow: hidden !important;
}
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0.5rem !important;
    max-width: 1180px;
    height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
}
div[data-testid="stVerticalBlock"] > div { gap: 0.4rem; }

/* Scrollable results well — only this area scrolls, never the page */
.mf-results-scroll {
    overflow-y: auto;
    flex: 1 1 auto;
    min-height: 0;
    padding-right: 4px;
}
.mf-results-scroll::-webkit-scrollbar { width: 7px; }
.mf-results-scroll::-webkit-scrollbar-thumb { background: var(--line); border-radius: 4px; }
.mf-results-scroll::-webkit-scrollbar-thumb:hover { background: var(--mint); }

/* ---------- Header ---------- */
.mf-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    color: var(--coral);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.mf-eyebrow::before {
    content: "";
    width: 6px; height: 6px;
    background: var(--coral);
    border-radius: 50%;
    display: inline-block;
}

h1.mf-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.7rem !important;
    line-height: 1.15 !important;
    letter-spacing: -0.01em;
    color: var(--ink) !important;
    margin: 0 0 0.25rem 0 !important;
}
.mf-title span { color: var(--mint); }

.mf-subtitle {
    font-size: 0.88rem;
    color: var(--muted);
    margin-bottom: 0.7rem;
    max-width: 640px;
}

/* ---------- Status strip ---------- */
.mf-status {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: var(--muted);
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    padding: 6px 2px;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 6px;
}
.mf-status b { color: var(--mint); font-weight: 600; }

/* ---------- Inputs ---------- */
.stTextInput > div > div > input {
    background: var(--surface) !important;
    color: var(--ink) !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 9px 14px !important;
}
.stTextInput > div > div > input::placeholder { color: var(--muted) !important; }
.stTextInput > div > div > input:focus {
    border-color: var(--coral) !important;
    box-shadow: 0 0 0 1px var(--coral) !important;
}

.stButton > button {
    background: var(--coral) !important;
    color: #11131a !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.01em;
    border: none !important;
    border-radius: 8px !important;
    padding: 9px 22px !important;
    width: 100%;
    transition: background .15s ease, transform .1s ease;
}
.stButton > button:hover { background: #ff7e5f !important; transform: translateY(-1px); }

.stSlider { padding-top: 0 !important; margin-top: -0.4rem; }
.stSlider label { color: var(--muted) !important; font-size: 0.78rem !important; }

/* ---------- Results heading ---------- */
.mf-results-heading {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.04em;
    margin: 0.5rem 0 0.5rem 0;
    text-transform: uppercase;
}
.mf-results-heading b { color: var(--ink); text-transform: none; }

/* ---------- Result card ---------- */
.mf-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 12px;
    transition: border-color .18s ease, transform .12s ease;
}
.mf-card:hover { border-color: var(--mint); transform: translateY(-2px); }

.mf-card-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 10px;
    border-bottom: 1px solid var(--line);
    background: var(--surface-2);
}
.mf-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.05em;
}
.mf-signal { display: flex; align-items: flex-end; gap: 2px; height: 13px; }
.mf-signal i {
    display: block; width: 4px; background: var(--line); border-radius: 1px;
}
.mf-signal i.on { background: var(--mint); }
.mf-signal i:nth-child(1) { height: 5px; }
.mf-signal i:nth-child(2) { height: 8px; }
.mf-signal i:nth-child(3) { height: 11px; }
.mf-signal i:nth-child(4) { height: 14px; }

.mf-card-body { padding: 8px 12px 10px; }

.mf-card-body img { max-height: 230px; object-fit: contain; }

.mf-meme-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--ink);
    margin: 7px 0 6px;
    line-height: 1.25;
}

.mf-tags { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 5px; }
.mf-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.02em;
    padding: 2px 7px;
    border-radius: 5px;
    background: var(--surface-2);
    border: 1px solid var(--line);
    color: var(--muted);
}
.mf-tag.cat { color: var(--coral); }
.mf-tag.fun { color: var(--mint); }

.mf-summary {
    font-size: 0.76rem;
    color: var(--muted);
    line-height: 1.4;
    margin-top: 2px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.mf-empty {
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    border: 1px dashed var(--line);
    border-radius: 8px;
    padding: 18px;
    text-align: center;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="mf-eyebrow">Semantic Meme Search · RAG Pipeline</div>', unsafe_allow_html=True)
st.markdown('<h1 class="mf-title">Find the meme you\'re <span>picturing</span>.</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="mf-subtitle">Describe a mood, a joke, or a vibe in plain English. '
    'Every meme is auto-tagged by a vision model and matched against your query by meaning, not keywords.</p>',
    unsafe_allow_html=True
)

# Build / load index
with st.spinner("Loading meme index…"):
    index, metadata = build_index(MEME_FOLDER)

if index is None:
    st.markdown(
        '<div class="mf-empty">No memes found. Add images to the <b>memes/</b> folder and reload.</div>',
        unsafe_allow_html=True
    )
    st.stop()

st.markdown(
    f'<div class="mf-status"><span><b>{len(metadata)}</b> memes indexed</span>'
    f'<span>embeddings · all-MiniLM-L6-v2</span>'
    f'<span>vector search · FAISS (cosine)</span>'
    f'<span>tagging · Groq llama-4-scout</span></div>',
    unsafe_allow_html=True
)

# Search bar
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "", placeholder='Try: "dark humor", "dad jokes", "relatable office memes"…',
        label_visibility="collapsed"
    )
with col2:
    search_btn = st.button("Search")

k = st.slider("Results to show", 3, 9, 6)

if (search_btn or query) and query.strip():
    results = search(query.strip(), index, metadata, k=k)

    if not results:
        st.markdown('<div class="mf-empty">No matches. Try a different phrase.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="mf-results-scroll">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="mf-results-heading">Top {len(results)} matches for <b>"{query}"</b></div>',
            unsafe_allow_html=True
        )

        cols = st.columns(3)
        for i, r in enumerate(results):
            with cols[i % 3]:
                st.markdown('<div class="mf-card">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="mf-card-head">'
                    f'<span class="mf-rank">RESULT {i+1:02d}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown('<div class="mf-card-body">', unsafe_allow_html=True)

                try:
                    img_name = Path(r["path"]).name
                    possible_paths = [
                        Path("memes") / img_name,
                        Path("/mount/src/meme-finder/memes") / img_name,
                        Path(r["path"]),
                    ]
                    found = False
                    for p in possible_paths:
                        if p.exists():
                            st.image(str(p), use_container_width=True)
                            found = True
                            break
                    if not found:
                        st.markdown(f'<div class="mf-empty">Image missing: {img_name}</div>', unsafe_allow_html=True)
                except Exception:
                    st.markdown('<div class="mf-empty">Image error</div>', unsafe_allow_html=True)

                d = r["data"]
                title     = d.get("title", "Untitled") or "Untitled"
                category  = d.get("category", "?") or "?"
                emotion   = d.get("emotion", "?") or "?"
                funniness = d.get("funniness", "?")
                summary   = d.get("summary", "") or ""

                st.markdown(f'<div class="mf-meme-title">{title}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="mf-tags">'
                    f'<span class="mf-tag cat">{category}</span>'
                    f'<span class="mf-tag">{emotion}</span>'
                    f'<span class="mf-tag fun">{funniness}/10 funny</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                if summary:
                    st.markdown(f'<div class="mf-summary">{summary}</div>', unsafe_allow_html=True)

                st.markdown('</div></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # close .mf-results-scroll
