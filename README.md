# 🐸 Meme Finder

A RAG-powered meme search engine. Search your meme collection using plain English.

**Live Demo → [meme-finder on Streamlit](https://memeragtest-f5qg5isskxf2ec6vt3kva4.streamlit.app/)**

---

## How it works

```
Image → Groq Vision LLM → FAISS Vector Index → Natural Language Search
```

## Features
- 🔍 Search by mood — *"dark humor"*, *"dad jokes"*, *"relatable"*
- 🏷️ Auto metadata — title, category, emotion, funniness score
- 🔁 Duplicate detection via image hashing
- 💾 Index saved to disk — only builds once

## Stack

| Tool | Purpose |
|------|---------|
| Groq LLM (llama-4-scout) | Analyze meme visually |
| FAISS | Vector similarity search |
| Sentence Transformers | Text embeddings |
| Streamlit | Web UI |

---

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/Muhammad-Shiraz/meme-finder
cd meme-finder
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your memes**
```
meme-finder/
└── memes/
    ├── meme1.jpg
    ├── meme2.png
    └── ...
```

**4. Add your Groq API key**

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```
Get a free key → [console.groq.com](https://console.groq.com)

**5. Install python-dotenv**
```bash
pip install python-dotenv
```

Add this at the top of `app.py`:
```python
from dotenv import load_dotenv
load_dotenv()
```

**6. Run**
```bash
streamlit run app.py
```

> First run builds the index (a few minutes). Every run after loads instantly from disk.

---

