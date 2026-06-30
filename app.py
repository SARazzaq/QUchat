"""
QUchat — Quran & Hadith Q&A

Search pipeline (state-of-the-art free stack):
  ┌─ Query Analysis ──────────────────────────────────────────────────────┐
  │  Islamic synonym expansion (instant, local dict, zero API cost)       │
  │  Porter stemming of all query terms                                   │
  └───────────────────────────────────────────────────────────────────────┘
  ┌─ Stage 1: BM25S Candidate Retrieval ──────────────────────────────────┐
  │  bm25s library — 500x faster than rank-bm25 via eager sparse scoring  │
  │  Pre-computed score matrix stored as scipy sparse — query in <50ms    │
  │  Corpus cached in /tmp after first load (no re-download on wake)      │
  └───────────────────────────────────────────────────────────────────────┘
  ┌─ Stage 2: TF-IDF Cosine Re-ranking ───────────────────────────────────┐
  │  sklearn TfidfVectorizer on top-100 BM25 candidates                   │
  │  Cosine similarity re-scores on full raw text (catches synonyms BM25  │
  │  may miss). Hybrid RRF (Reciprocal Rank Fusion) combines both scores. │
  └───────────────────────────────────────────────────────────────────────┘
  ┌─ Stage 3: Answer Generation ──────────────────────────────────────────┐
  │  Groq Llama 3.3 70B — streaming, free tier                            │
  │  Full citation: Surah name, Arabic name, chapter, verse, narrator     │
  └───────────────────────────────────────────────────────────────────────┘
"""

import re, gzip, json, string, os, threading, time
import numpy as np
import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

import bm25s
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()

# ── Hardcoded stopwords (no NLTK corpus download needed) ─────────────────────
_SW = {
    "i","me","my","we","our","you","your","he","him","his","she","her","it",
    "its","they","them","their","what","which","who","this","that","these",
    "those","am","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","a","an","the","and","but","if","or","as","of",
    "at","by","for","with","about","to","from","in","out","on","so","than",
    "too","very","can","will","just","now","s","t","d","ll","m","re","ve",
    "said","also","upon","unto","thee","thy","thou","shall","hath","ye","lo",
}

# ── Islamic synonym dictionary ────────────────────────────────────────────────
_SYN: dict[str, list[str]] = {
    "namaz":       ["salah","salat","prayer","worship","prostrate","bow","establish","iqama"],
    "salah":       ["namaz","salat","prayer","worship","prostrate","bow","establish"],
    "prayer":      ["salah","salat","namaz","worship","supplication","dua","invoke","pray"],
    "roza":        ["fasting","sawm","fast","ramadan","abstain","iftar","suhoor"],
    "fasting":     ["sawm","roza","fast","ramadan","abstain","hunger","thirst"],
    "zakat":       ["alms","charity","zakah","tithe","poor","needy","purification","give"],
    "hajj":        ["pilgrimage","makkah","kaaba","tawaf","ihram","arafat","umrah"],
    "jihad":       ["struggle","striving","effort","fight","path","sabeel"],
    "death":       ["die","dying","dead","deceased","perish","mawt","wafat","ruh","soul",
                    "grave","barzakh","akhirah","afterlife","qiyamah","resurrection",
                    "judgment","angel","israfil","trumpet","malak","azrael"],
    "die":         ["death","dead","dying","perish","mawt","soul","grave","wafat"],
    "soul":        ["ruh","spirit","nafs","death","barzakh","grave","life"],
    "grave":       ["barzakh","tomb","burial","death","questioning","munkar","nakir","qabr"],
    "barzakh":     ["grave","intermediate","death","soul","afterlife","barrier"],
    "akhirah":     ["afterlife","hereafter","qiyamah","resurrection","judgment",
                    "paradise","hell","death","next","world"],
    "resurrection":["qiyamah","judgment","akhirah","day","trumpet","israfil","rise","dead"],
    "judgment":    ["qiyamah","resurrection","akhirah","scales","deeds","account","reckoning"],
    "paradise":    ["jannah","heaven","garden","reward","bliss","afterlife","eternal"],
    "jannah":      ["paradise","heaven","garden","reward","bliss","houri","rivers"],
    "hell":        ["jahannam","fire","punishment","torment","akhirah","wrath","azab"],
    "jahannam":    ["hell","fire","punishment","torment","wrath","azab"],
    "haram":       ["forbidden","prohibited","unlawful","sin","transgression","avoid"],
    "halal":       ["permissible","lawful","allowed","permitted","pure","clean"],
    "sin":         ["haram","transgression","evil","wrong","disobey","gunah","ithm"],
    "tawbah":      ["repentance","forgiveness","mercy","return","regret","istighfar"],
    "repentance":  ["tawbah","forgiveness","mercy","return","regret","istighfar","turn"],
    "alcohol":     ["wine","khamr","intoxicant","drinking","liquor","forbidden","haram"],
    "riba":        ["interest","usury","forbidden","loan","bank","excess"],
    "marriage":    ["nikah","wed","spouse","husband","wife","contract","mahr","walima"],
    "divorce":     ["talaq","separation","marriage","iddah","khul"],
    "parents":     ["mother","father","obedience","honor","respect","family","birr"],
    "women":       ["female","hijab","modesty","wife","mother","sister","awrah"],
    "hijab":       ["veil","cover","modesty","women","purdah","awrah","dress"],
    "knowledge":   ["ilm","learn","wisdom","education","scholar","seek","study"],
    "patience":    ["sabr","endure","trial","hardship","steadfast","persevere"],
    "charity":     ["sadaqah","zakat","give","poor","needy","help","infaq"],
    "justice":     ["adl","fairness","equity","rights","balance","qist"],
    "pork":        ["pig","swine","forbidden","haram","meat","food"],
    "food":        ["eat","halal","haram","meat","pork","pig","drink","tayyib"],
    "prophet":     ["muhammad","messenger","rasool","nabi","sunnah","pbuh"],
    "muhammad":    ["prophet","messenger","rasool","nabi","sunnah"],
    "allah":       ["god","lord","creator","rabb","divine","worship","ilah"],
    "quran":       ["revelation","scripture","book","ayah","verse","surah","furqan"],
    "obligatory":  ["fard","wajib","mandatory","compulsory","duty","required"],
    "fard":        ["obligatory","wajib","mandatory","compulsory","duty"],
    "sunnah":      ["prophet","recommended","voluntary","practice","hadith"],
    "dua":         ["prayer","supplication","invoke","ask","request","call"],
    "taqwa":       ["piety","fear","god","consciousness","righteousness"],
    "iman":        ["faith","belief","trust","conviction","islam"],
    "shirk":       ["polytheism","association","idol","partner","worship"],
    "tawhid":      ["monotheism","oneness","unity","god","allah"],
}

def expand_query(question: str) -> list[str]:
    """Instant local expansion using Islamic synonym dict + stemming."""
    q = question.lower()
    tokens = [w.strip(string.punctuation) for w in q.split()
              if len(w.strip(string.punctuation)) > 2]
    terms: set[str] = set(tokens)
    for t in tokens:
        terms.update(_SYN.get(t, []))
        st_t = _stemmer.stem(t)
        for k, v in _SYN.items():
            if _stemmer.stem(k) == st_t:
                terms.update(v)
    # bigrams
    for i in range(len(tokens)-1):
        terms.add(f"{tokens[i]} {tokens[i+1]}")
    return list(terms)

# ── Fast normaliser ───────────────────────────────────────────────────────────
def normalise(text: str) -> str:
    """Return stemmed, stopword-filtered string for BM25S tokeniser."""
    out = []
    for t in text.lower().split():
        t = t.strip(string.punctuation)
        if t and t.isalpha() and t not in _SW:
            out.append(_stemmer.stem(t))
    return " ".join(out)

# ── Endpoints ─────────────────────────────────────────────────────────────────
QURAN_URL    = "https://api.alquran.cloud/v1/quran/en.sahih"
HADITH_CDN   = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/{b}.json"
HADITH_BOOKS = {
    "Sahih Bukhari":    "eng-bukhari",
    "Sahih Muslim":     "eng-muslim",
    "Sunan Abu Dawud":  "eng-abudawud",
    "Jami at-Tirmidhi": "eng-tirmidhi",
}
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
CORPUS_CACHE = "/tmp/quchat_v3.json.gz"

# ══════════════════════════════════════════════════════════════════════════════
# CORPUS BUILD + INDEX (cached in /tmp — built once per deployment)
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_quran() -> list[dict]:
    r = requests.get(QURAN_URL, timeout=40)
    r.raise_for_status()
    docs = []
    for s in r.json()["data"]["surahs"]:
        for a in s["ayahs"]:
            docs.append({
                "t":"q",
                "sn": s["number"],           # surah number
                "se": s["englishName"],       # english name
                "sa": s["name"],              # arabic name
                "an": a["numberInSurah"],     # ayah number
                "tx": a["text"],              # full english text
            })
    return docs

def _fetch_hadith_book(bn: str, bi: str) -> list[dict]:
    try:
        r = requests.get(HADITH_CDN.format(b=bi), timeout=40)
        r.raise_for_status()
        return [{"t":"h","bk":bn,"no":str(h.get("hadithnumber","")),
                 "nr":h.get("by",""),"tx":h.get("text","")}
                for h in r.json().get("hadiths",[]) if h.get("text","")]
    except Exception:
        return []

@st.cache_resource(show_spinner=False)
def build_index():
    """
    Returns (docs, bm25_retriever, tfidf_matrix, tfidf_vectorizer, norm_texts)
    - Loads from /tmp cache if available (instant on re-deploy)
    - Otherwise fetches all sources in parallel, normalises, indexes
    """
    # ── Try disk cache ────────────────────────────────────────────────────────
    if os.path.exists(CORPUS_CACHE):
        try:
            with gzip.open(CORPUS_CACHE, "rt", encoding="utf-8") as f:
                docs = json.load(f)
            norm_texts = [d.get("nk","") or normalise(d["tx"]) for d in docs]
            retriever  = bm25s.BM25()
            retriever.index(bm25s.tokenize(norm_texts, stopwords=None))
            vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2,
                                         max_features=80000, sublinear_tf=True)
            tfidf_mat  = vectorizer.fit_transform([d["tx"] for d in docs])
            return docs, retriever, tfidf_mat, vectorizer, norm_texts
        except Exception:
            pass

    # ── Parallel fetch ────────────────────────────────────────────────────────
    q_docs: list[dict] = []
    h_docs: list[dict] = []

    def _fq():
        nonlocal q_docs
        try: q_docs = _fetch_quran()
        except Exception: pass

    def _fh():
        nonlocal h_docs
        with ThreadPoolExecutor(max_workers=4) as p:
            for res in p.map(lambda x: _fetch_hadith_book(*x), HADITH_BOOKS.items()):
                h_docs.extend(res)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_fq)
        f2 = pool.submit(_fh)
        f1.result(); f2.result()

    all_docs = q_docs + h_docs
    if not all_docs:
        return [], None, None, None, []

    # ── Parallel normalisation ────────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=8) as pool:
        norm_texts = list(pool.map(normalise, [d["tx"] for d in all_docs]))

    for d, nk in zip(all_docs, norm_texts):
        d["nk"] = nk

    # ── BM25S index (eager sparse scoring — 500x faster than rank-bm25) ──────
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(norm_texts, stopwords=None))

    # ── TF-IDF matrix (for cosine re-ranking stage) ───────────────────────────
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2,
                                  max_features=80000, sublinear_tf=True)
    tfidf_mat  = vectorizer.fit_transform([d["tx"] for d in all_docs])

    # ── Save to /tmp for instant reload ──────────────────────────────────────
    try:
        with gzip.open(CORPUS_CACHE, "wt", encoding="utf-8", compresslevel=4) as f:
            json.dump(all_docs, f, ensure_ascii=False, separators=(",",":"))
    except Exception:
        pass

    return all_docs, retriever, tfidf_mat, vectorizer, norm_texts


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVAL: BM25S + TF-IDF cosine + Reciprocal Rank Fusion
# ══════════════════════════════════════════════════════════════════════════════
def _rrf(rank_a: list[int], rank_b: list[int], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion — combines two ranked lists into one score."""
    scores: dict[int, float] = {}
    for rank, idx in enumerate(rank_a):
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(rank_b):
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
    return scores

def search(question: str, top_n: int = 15) -> tuple[list[dict], list[dict], list[str], float]:
    """
    Full hybrid pipeline:
      1. BM25S: retrieve top-200 candidates (sparse, vectorised)
      2. TF-IDF cosine: re-rank those 200 candidates
      3. RRF: fuse BM25S ranks + cosine ranks
      4. Split into Quran / Hadith, return top_n of each
    """
    docs, retriever, tfidf_mat, vectorizer, _ = build_index()
    if not docs or retriever is None:
        return [], [], [], 0.0

    t0 = time.time()
    raw_terms = expand_query(question)

    # ── Stage 1: BM25S retrieval ─────────────────────────────────────────────
    query_norm = normalise(question + " " + " ".join(raw_terms))
    tokenized_q = bm25s.tokenize([query_norm], stopwords=None)
    results, scores = retriever.retrieve(tokenized_q, k=min(200, len(docs)))
    bm25_indices = results[0].tolist()   # indices of top-200 docs

    # ── Stage 2: TF-IDF cosine on top-200 candidates ─────────────────────────
    q_vec       = vectorizer.transform([question + " " + " ".join(raw_terms)])
    cand_matrix = tfidf_mat[bm25_indices]
    cos_scores  = cosine_similarity(q_vec, cand_matrix)[0]
    cosine_rank = [bm25_indices[i] for i in np.argsort(-cos_scores)]

    # ── Stage 3: Reciprocal Rank Fusion ──────────────────────────────────────
    rrf_scores  = _rrf(bm25_indices, cosine_rank)
    fused       = sorted(rrf_scores.keys(), key=lambda i: -rrf_scores[i])

    elapsed = time.time() - t0

    # ── Split and return ──────────────────────────────────────────────────────
    q_hits = [docs[i] for i in fused if docs[i]["t"] == "q"][:top_n]
    h_hits = [docs[i] for i in fused if docs[i]["t"] == "h"][:top_n]

    return q_hits, h_hits, raw_terms, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT HELPERS — rich citations
# ══════════════════════════════════════════════════════════════════════════════
def fmt_q_display(vv: list[dict]) -> str:
    return "\n\n".join(
        f"[Surah {v['se']} ({v['sa']}) — Chapter {v['sn']}, Verse {v['an']}]\n{v['tx']}"
        for v in vv)

def fmt_h_display(hh: list[dict]) -> str:
    lines = []
    for h in hh:
        nar = f" | Narrated by: {h['nr']}" if h["nr"] else ""
        lines.append(f"[{h['bk']}, Hadith #{h['no']}{nar}]\n{h['tx']}")
    return "\n\n".join(lines)

def fmt_q_llm(vv: list[dict]) -> str:
    """Rich citation format for LLM context."""
    return "\n\n".join(
        f"SOURCE Q{i+1}: Surah {v['se']} ({v['sa']}) — "
        f"Chapter {v['sn']}, Verse {v['an']}\n"
        f'"{v["tx"]}"'
        for i, v in enumerate(vv))

def fmt_h_llm(hh: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hh):
        nar  = f"Narrated by: {h['nr']} — " if h["nr"] else ""
        text = h["tx"][:550]+"…" if len(h["tx"])>550 else h["tx"]
        lines.append(
            f"SOURCE H{i+1}: {h['bk']}, Hadith #{h['no']} — {nar}\n\"{text}\""
        )
    return "\n\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — strict citation format
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are a deeply knowledgeable Islamic scholar. Answer using ONLY the numbered sources provided (Q1, Q2… for Quran; H1, H2… for Hadith). Never fabricate or add outside knowledge.

MANDATORY ANSWER FORMAT:

---

## 📖 From the Quran

For EVERY Quran source (Q1, Q2…):

**[Surah {English Name} ({Arabic Name}) — Chapter X, Verse Y]**
> *[Paste the COMPLETE verbatim text of the verse exactly as given]*

**Explanation:** 3-5 sentences — what Allah ﷻ commands/informs here, its direct relevance to the question, and its theological significance.

---

## 📜 From the Hadith

For EVERY Hadith source (H1, H2…):

**[{Book Name}, Hadith #{Number} — Narrated by: {Narrator} (رضي الله عنه/عنها)]**
> *[Paste the COMPLETE verbatim text of the Hadith exactly as given]*

**Explanation:** 3-5 sentences — what the Prophet ﷺ said/did, the ruling it establishes, and its relevance to the question.

---

## ✅ Islamic Ruling & Summary

A comprehensive paragraph that:
1. Directly answers the question
2. States the ruling (Fard فرض / Sunnah سنة / Haram حرام / Mubah مباح) if applicable
3. Cites every source inline: (Surah Al-Baqarah 2:286), (Sahih Bukhari #1386)
4. Explains the Islamic wisdom behind the ruling

---

ABSOLUTE RULES:
- ﷺ after EVERY mention of Prophet Muhammad
- رضي الله عنه / رضي الله عنها after EVERY Companion name
- Quote every source verbatim in blockquote BEFORE explaining
- Cover ALL provided sources — skip none
- Never invent content not in the provided sources"""

def stream_answer(question: str, q_docs: list[dict], h_docs: list[dict]):
    """Stream Groq response token by token."""
    key = st.secrets.get("GROQ_API_KEY","")
    if not key:
        yield "⚠️ Groq API key not configured. Add GROQ_API_KEY to Streamlit secrets."
        return

    user_msg = (
        f"QUESTION: {question}\n\n"
        f"=== QURAN SOURCES (Sahih International) ===\n"
        f"{fmt_q_llm(q_docs) if q_docs else 'None found.'}\n\n"
        f"=== HADITH SOURCES (Bukhari / Muslim / Abu Dawud / Tirmidhi) ===\n"
        f"{fmt_h_llm(h_docs) if h_docs else 'None found.'}\n\n"
        f"Write the complete, fully cited scholarly answer following the mandatory format exactly."
    )
    try:
        r = requests.post(
            GROQ_URL,
            json={"model": GROQ_MODEL,
                  "messages": [{"role":"system","content":SYSTEM_PROMPT},
                                {"role":"user",  "content":user_msg}],
                  "temperature": 0.1, "max_tokens": 3500, "stream": True},
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            stream=True, timeout=90,
        )
        r.raise_for_status()
        for line in r.iter_lines():
            if not line: continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]": break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content","")
                    if delta: yield delta
                except Exception: continue
    except Exception as e:
        yield f"\n\nError: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="QUchat", page_icon="🕌", layout="centered")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;color:#f0ece4;}
.stApp{background:#0f1e0f;}
#MainMenu,footer,header{visibility:hidden;}
.stChatInput textarea{background:#1a2e1a!important;color:#f5f0e8!important;
  border:1.5px solid #b8960c!important;border-radius:14px!important;
  font-family:'Inter',sans-serif!important;font-size:1rem!important;}
.stChatInput textarea::placeholder{color:#7a9a7a!important;}
.stChatMessage{background:transparent!important;}
[data-testid="stChatMessageContent"]{font-family:'Inter',sans-serif;
  font-size:1rem;line-height:1.9;color:#f0ece4;}
details{background:#152015!important;border:1px solid #2a4a2a!important;
  border-radius:10px!important;margin-bottom:6px;}
summary{color:#d4a017!important;font-weight:600!important;
  font-size:0.88rem!important;padding:10px 14px!important;}
.streamlit-expanderContent{color:#c8dcc0!important;font-size:0.82rem!important;
  font-family:'Inter',sans-serif!important;line-height:1.6!important;
  white-space:pre-wrap!important;}
[data-testid="stAlert"]{background:#1a2a1a!important;
  border-left:4px solid #d4a017!important;border-radius:10px!important;
  color:#e8dfc8!important;font-size:0.9rem!important;}
.stSpinner>div{border-top-color:#d4a017!important;}
::-webkit-scrollbar{width:5px;}::-webkit-scrollbar-track{background:#0f1e0f;}
::-webkit-scrollbar-thumb{background:#b8960c;border-radius:3px;}
.arabic{font-family:'Amiri',serif;font-size:1.4rem;color:#f5e6a0;direction:rtl;line-height:2.4;}
hr{border-color:#1e3a1e!important;}
a{color:#7ec88a!important;}a:hover{color:#d4a017!important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:2.5rem 0 1.2rem 0;">
  <div style="font-size:3.2rem;margin-bottom:0.3rem;">🕌</div>
  <h1 style="font-family:'Amiri',serif;color:#d4a017;font-size:2.8rem;margin:0;letter-spacing:1px;">QUchat</h1>
  <p style="color:#7ec88a;font-size:1rem;margin:0.4rem 0 0.8rem 0;letter-spacing:2px;text-transform:uppercase;font-weight:500;">Quran & Hadith Question Answering</p>
  <p class="arabic">بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ</p>
  <p style="color:#8aaa88;font-size:0.85rem;font-style:italic;margin-top:0.2rem;">In the name of Allah, the Most Gracious, the Most Merciful</p>
</div><hr>
""", unsafe_allow_html=True)

st.info(
    "⚠️ **Disclaimer:** Answers are AI-generated from authentic Islamic sources. "
    "Always verify with a qualified Islamic scholar. "
    "Quran: Sahih International · Hadith: Bukhari, Muslim, Abu Dawud, Tirmidhi."
)

# ── Silent background warm-up ─────────────────────────────────────────────────
if "warmed" not in st.session_state:
    st.session_state["warmed"] = True
    threading.Thread(target=build_index, daemon=True).start()

# ══════════════════════════════════════════════════════════════════════════════
# CHAT UI
# ══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about the Quran or Islam…"):
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.status("🔍 Searching…", expanded=True) as status:

            st.write("⚡ **Stage 1** — BM25S sparse retrieval (top-200 candidates)…")
            st.write("⚡ **Stage 2** — TF-IDF cosine re-ranking on candidates…")
            st.write("⚡ **Stage 3** — Reciprocal Rank Fusion (BM25S + cosine)…")

            q_hits, h_hits, terms, elapsed = search(user_input, top_n=15)

            st.write(f"✅ **Search complete in {elapsed:.3f}s**")
            st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ {len(terms)} query terms · "
                     f"200 BM25S candidates · TF-IDF re-ranked · RRF fused")

            if q_hits:
                st.write(f"📖 **{len(q_hits)} Quran verses** ranked — top 8 → answer:")
                for i, v in enumerate(q_hits[:5], 1):
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;**#{i}** — "
                             f"Surah *{v['se']}* ({v['sa']}) {v['sn']}:{v['an']}")
                if len(q_hits) > 5:
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;…+{len(q_hits)-5} more")
            else:
                st.write("⚠️ No Quran verses matched.")

            if h_hits:
                st.write(f"📜 **{len(h_hits)} Hadiths** ranked — top 6 → answer:")
                for i, h in enumerate(h_hits[:5], 1):
                    nar = f" · {h['nr']}" if h["nr"] else ""
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;**#{i}** — "
                             f"*{h['bk']}* #{h['no']}{nar}")
                if len(h_hits) > 5:
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;…+{len(h_hits)-5} more")
            else:
                st.write("⚠️ No Hadiths matched.")

            if not q_hits and not h_hits:
                status.update(label="❌ No results found", state="error")
                answer = "No relevant sources found. Please rephrase your question."
                st.markdown(answer)
                st.session_state.messages.append({"role":"assistant","content":answer})
            else:
                n = min(8,len(q_hits)) + min(6,len(h_hits))
                status.update(
                    label=f"✅ {len(q_hits)} verses · {len(h_hits)} hadiths · "
                          f"{n} cited · {elapsed:.3f}s search",
                    state="complete", expanded=False
                )

        if q_hits:
            with st.expander(f"📖 {len(q_hits)} Quran verses (BM25S+TF-IDF ranked)"):
                st.text(fmt_q_display(q_hits))
        if h_hits:
            with st.expander(f"📜 {len(h_hits)} Hadiths (BM25S+TF-IDF ranked)"):
                st.text(fmt_h_display(h_hits))

        if q_hits or h_hits:
            full = ""
            placeholder = st.empty()
            for chunk in stream_answer(user_input, q_hits[:8], h_hits[:6]):
                full += chunk
                placeholder.markdown(full + "▌")
            placeholder.markdown(full)
            st.session_state.messages.append({"role":"assistant","content":full})

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<p style="text-align:center;color:#4a7a4a;font-size:0.78rem;margin:0.6rem 0 0.2rem 0;">
  Quran · <a href="https://alquran.cloud">alquran.cloud</a> (Sahih International, 6,236 verses) &nbsp;·&nbsp;
  Hadith · <a href="https://github.com/fawazahmed0/hadith-api">fawazahmed0/hadith-api</a> (Bukhari · Muslim · Abu Dawud · Tirmidhi) &nbsp;·&nbsp;
  Search · <a href="https://bm25s.github.io">BM25S</a> + TF-IDF cosine + RRF &nbsp;·&nbsp;
  LLM · <a href="https://groq.com">Groq</a> Llama 3.3 70B (streaming) &nbsp;·&nbsp;
  UI · <a href="https://streamlit.io">Streamlit</a>
</p>
<p style="text-align:center;color:#2a4a2a;font-size:0.72rem;margin:0 0 1rem 0;">
  All sources free &amp; open. No user data stored.
</p>
""", unsafe_allow_html=True)
