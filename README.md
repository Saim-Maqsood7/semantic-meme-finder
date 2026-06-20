# sementic meme finder

A RAG-powered meme search engine. Search your meme collection using plain English.

# You can check it out! it is live
**live demo** = https://semantic-meme-finder-lmxpykk82q5z3fdzrswycg.streamlit.app/

# How it works
```
Image → Groq Vision LLM → FAISS Vector Index → Natural Language Search
```
# Features
- 🔍 Search by mood — *"dark humor"*, *"dad jokes"*, *"relatable"*
- 🏷️ Auto metadata — title, category, emotion, funniness score
- 🔁 Duplicate detection via image hashing
- 💾 Index saved to disk — only builds once

# Stack
| Tool | Purpose |
|------|---------|
| Groq LLM (llama-4-scout) | Analyze meme visually |
| FAISS | Vector similarity search |
| Sentence Transformers | Text embeddings |
| Streamlit | Web UI |
---
# Run Locally
**1. Clone the repo**
```bash
git clone https://github.com/Saim-Maqsood7/semantic-meme-finder
cd sementic-meme-finder
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

