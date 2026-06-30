"""
QUchat — Quran & Hadith Q&A  (optimised parallel pipeline)

Speed optimisations:
  - LLM query expansion + Quran index build + Hadith index build run IN PARALLEL
  - Fast NLP: simple lemmatiser (no per-token POS tagging), Porter stemmer
  - Batch normalisation with ThreadPoolExecutor
  - @st.cache_resource: indices built once, reused across all sessions
  - BM25Okapi scores entire corpus in a single vectorised pass
"""

import re, string, threading
import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── NLP ───────────────────────────────────────────────────────────────────────
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi

for _p in ("punkt", "wordnet", "stopwords", "punkt_tab", "omw-1.4"):
    try:
        nltk.download(_p, quiet=True)
    except Exception:
        pass

_stemmer    = PorterStemmer()
_lemmatizer = WordNetLemmatizer()
_stopwords  = set(stopwords.words("english"))

# ── Endpoints ─────────────────────────────────────────────────────────────────
QURAN_URL  = "https://api.alquran.cloud/v1/quran/en.sahih"
HADITH_CDN = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/{b}.json"
HADITH_BOOKS = {
    "Sahih Bukhari":    "eng-bukhari",
    "Sahih Muslim":     "eng-muslim",
    "Sunan Abu Dawud":  "eng-abudawud",
    "Jami at-Tirmidhi": "eng-tirmidhi",
}
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

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
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:#0f1e0f;}
::-webkit-scrollbar-thumb{background:#b8960c;border-radius:3px;}
.arabic{font-family:'Amiri',serif;font-size:1.4rem;color:#f5e6a0;
  direction:rtl;line-height:2.4;}
hr{border-color:#1e3a1e!important;}
a{color:#7ec88a!important;}a:hover{color:#d4a017!important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:2.5rem 0 1.2rem 0;">
  <div style="font-size:3.2rem;margin-bottom:0.3rem;">🕌</div>
  <h1 style="font-family:'Amiri',serif;color:#d4a017;font-size:2.8rem;margin:0;
      letter-spacing:1px;">QUchat</h1>
  <p style="color:#7ec88a;font-size:1rem;margin:0.4rem 0 0.8rem 0;
      letter-spacing:2px;text-transform:uppercase;font-weight:500;">
    Quran & Hadith Question Answering</p>
  <p class="arabic">بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ</p>
  <p style="color:#8aaa88;font-size:0.85rem;font-style:italic;margin-top:0.2rem;">
    In the name of Allah, the Most Gracious, the Most Merciful</p>
</div><hr>
""", unsafe_allow_html=True)

st.info(
    "⚠️ **Disclaimer:** Answers are AI-generated from authentic Islamic sources. "
    "Always verify with a qualified Islamic scholar. "
    "Quran: Sahih International · Hadith: Bukhari, Muslim, Abu Dawud, Tirmidhi."
)

# ══════════════════════════════════════════════════════════════════════════════
# FAST NLP — no per-token POS tagging (10x faster than POS-aware version)
# ══════════════════════════════════════════════════════════════════════════════

def normalise(text: str) -> list[str]:
    """
    Fast pipeline: lowercase → split → strip punctuation → drop stopwords
    → lemmatise as noun (fast, good enough for IR) → stem.
    Avoids NLTK pos_tag which is the main bottleneck.
    """
    out = []
    for t in text.lower().split():
        t = t.strip(string.punctuation)
        if not t or not t.isalpha() or t in _stopwords:
            continue
        out.append(_stemmer.stem(_lemmatizer.lemmatize(t)))
    return out


def batch_normalise(texts: list[str]) -> list[list[str]]:
    """Parallelise normalisation across CPU threads."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(normalise, texts))


# ══════════════════════════════════════════════════════════════════════════════
# GROQ HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _groq(messages: list[dict], max_tokens: int = 600, temp: float = 0.1) -> str:
    key = st.secrets.get("GROQ_API_KEY", "")
    if not key:
        return ""
    r = requests.post(
        GROQ_URL,
        json={"model": GROQ_MODEL, "messages": messages,
              "temperature": temp, "max_tokens": max_tokens},
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        timeout=45,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ══════════════════════════════════════════════════════════════════════════════
# INDEX BUILDING — cached, parallel, batched
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def build_quran_index():
    """Fetch + batch-normalise all 6,236 verses, build BM25 index."""
    try:
        r = requests.get(QURAN_URL, timeout=40)
        r.raise_for_status()
        verses = []
        raw_texts = []
        for s in r.json().get("data", {}).get("surahs", []):
            for a in s.get("ayahs", []):
                verses.append({
                    "surah_no":   s["number"],
                    "surah_name": s["englishName"],
                    "surah_arab": s["name"],
                    "ayah_no":    a["numberInSurah"],
                    "text":       a["text"],
                })
                raw_texts.append(a["text"])
        # Batch normalise in parallel
        corpus = batch_normalise(raw_texts)
        return verses, BM25Okapi(corpus)
    except Exception:
        return [], None


def _fetch_one_hadith_book(book_name: str, book_id: str) -> list[dict]:
    try:
        r = requests.get(HADITH_CDN.format(b=book_id), timeout=40)
        r.raise_for_status()
        return [
            {"book": book_name,
             "number": h.get("hadithnumber", ""),
             "narrator": h.get("by", ""),
             "text": h.get("text", "")}
            for h in r.json().get("hadiths", [])
            if h.get("text", "")
        ]
    except Exception:
        return []


@st.cache_resource(show_spinner=False)
def build_hadith_index():
    """
    Fetch all 4 hadith books IN PARALLEL, batch-normalise all texts,
    build one unified BM25 index.
    """
    all_hadiths: list[dict] = []

    # Download all 4 books simultaneously
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_fetch_one_hadith_book, bn, bi): bn
            for bn, bi in HADITH_BOOKS.items()
        }
        for fut in as_completed(futures):
            all_hadiths.extend(fut.result())

    if not all_hadiths:
        return [], None

    # Batch normalise all hadith texts in parallel
    raw_texts = [h["text"] for h in all_hadiths]
    corpus    = batch_normalise(raw_texts)
    return all_hadiths, BM25Okapi(corpus)


# ══════════════════════════════════════════════════════════════════════════════
# QUERY EXPANSION — LLM generates 35+ Islamic search terms
# ══════════════════════════════════════════════════════════════════════════════

def generate_search_terms(question: str) -> list[str]:
    prompt = f"""You are an expert Islamic scholar preparing an exhaustive Quran and Hadith search.

Produce search terms for the question covering:
1. Core subject (English)
2. Arabic/transliterated Islamic terms (mawt, barzakh, akhirah, qiyamah, ruh…)
3. Synonyms (die, dying, dead, deceased, perish…)
4. Connected concepts (soul, grave, resurrection, judgment, paradise, hell, angels…)
5. Quranic themes, associated Prophets/figures
6. Events (burial, questioning in grave, trumpet blow, scales of deeds)
7. Opposites (life, dunya) and consequences (reward, punishment)

Question: {question}

Return ONLY a Python list of 35-45 lowercase strings. No explanations.
["death", "die", "soul", "grave", "barzakh", ...]"""
    try:
        raw   = _groq([{"role": "user", "content": prompt}], max_tokens=500, temp=0.3)
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            terms = eval(match.group())
            if isinstance(terms, list):
                return [str(t).strip().lower() for t in terms
                        if t and len(str(t).strip()) > 1][:45]
    except Exception:
        pass
    return [w.strip("?.,!;:'\"").lower() for w in question.split() if len(w) > 2]


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL — BM25 + phrase boost
# ══════════════════════════════════════════════════════════════════════════════

def _query_tokens(raw_terms: list[str]) -> list[str]:
    tokens: list[str] = []
    for t in raw_terms:
        tokens.extend(normalise(t))
    seen: set[str] = set()
    return [x for x in tokens if not (x in seen or seen.add(x))]  # type: ignore


def _phrase_boost(text_lower: str, raw_terms: list[str]) -> float:
    return sum(3.0 for t in raw_terms if " " in t and t in text_lower)


def retrieve_quran(raw_terms: list[str], top_n: int = 15) -> list[dict]:
    verses, bm25 = build_quran_index()
    if not verses or bm25 is None:
        return []
    qtoks = _query_tokens(raw_terms)
    if not qtoks:
        return []
    scores = bm25.get_scores(qtoks).tolist()
    for i, v in enumerate(verses):
        scores[i] += _phrase_boost(v["text"].lower(), raw_terms)
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
    return [verses[i] for i in top if scores[i] > 0]


def retrieve_hadith(raw_terms: list[str], top_n: int = 15) -> list[dict]:
    hadiths, bm25 = build_hadith_index()
    if not hadiths or bm25 is None:
        return []
    qtoks = _query_tokens(raw_terms)
    if not qtoks:
        return []
    scores = bm25.get_scores(qtoks).tolist()
    for i, h in enumerate(hadiths):
        scores[i] += _phrase_boost(h["text"].lower(), raw_terms)
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
    return [hadiths[i] for i in top if scores[i] > 0]


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fmt_q_display(vv: list[dict]) -> str:
    return "\n\n".join(
        f"[{v['surah_name']} ({v['surah_arab']}) {v['surah_no']}:{v['ayah_no']}]\n{v['text']}"
        for v in vv)

def fmt_h_display(hh: list[dict]) -> str:
    lines = []
    for h in hh:
        nar = f" | {h['narrator']}" if h["narrator"] else ""
        lines.append(f"[{h['book']}, #{h['number']}{nar}]\n{h['text']}")
    return "\n\n".join(lines)

def fmt_q_llm(vv: list[dict]) -> str:
    return "\n\n".join(
        f'Surah {v["surah_name"]} ({v["surah_arab"]}) — '
        f'Ch.{v["surah_no"]}, V.{v["ayah_no"]}:\n"{v["text"]}"'
        for v in vv)

def fmt_h_llm(hh: list[dict]) -> str:
    lines = []
    for h in hh:
        nar  = f" — Narrated by: {h['narrator']}" if h["narrator"] else ""
        text = h["text"][:500] + "…" if len(h["text"]) > 500 else h["text"]
        lines.append(f'{h["book"]}, Hadith #{h["number"]}{nar}:\n"{text}"')
    return "\n\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# ANSWER GENERATION
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a deeply knowledgeable Islamic scholar. Answer using ONLY the Quran verses and Hadiths provided — never fabricate or add external knowledge.

ANSWER FORMAT:

## 📖 Quranic Evidence

For each verse:
**Surah [Name] [Chapter:Verse]**
> *[Quote the verse verbatim in full]*
**Explanation:** 2-4 sentences on what this verse says about the question.

## 📜 Hadith Evidence

For each hadith:
**[Book], Hadith #[N] — Narrated by [Name] (رضي الله عنه/عنها)**
> *[Quote the hadith verbatim in full]*
**Explanation:** 2-4 sentences on its ruling, context, and relevance.

## ✅ Summary & Islamic Ruling

One comprehensive paragraph with inline citations like (Al-Baqarah 2:155) and (Sahih Bukhari #1386).

RULES: Quote every source verbatim first. ﷺ after Prophet ﷺ always. Cover ALL sources. Never fabricate."""


def ask_groq(question: str, q_llm: list[dict], h_llm: list[dict]) -> str:
    user_msg = (
        f"Question: {question}\n\n"
        f"=== QURAN (Sahih International) ===\n{fmt_q_llm(q_llm) if q_llm else 'None.'}\n\n"
        f"=== HADITH ===\n{fmt_h_llm(h_llm) if h_llm else 'None.'}\n\n"
        f"Write the complete cited scholarly answer following the mandatory format."
    )
    result = _groq(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user",   "content": user_msg}],
        max_tokens=3000, temp=0.1,
    )
    return result or "⚠️ Groq API key not configured. Add GROQ_API_KEY to Streamlit secrets."


# ══════════════════════════════════════════════════════════════════════════════
# SILENT BACKGROUND WARM-UP (non-blocking — fires once per deployment)
# ══════════════════════════════════════════════════════════════════════════════
if "warmed" not in st.session_state:
    st.session_state["warmed"] = True
    threading.Thread(
        target=lambda: (build_quran_index(), build_hadith_index()),
        daemon=True
    ).start()


# ══════════════════════════════════════════════════════════════════════════════
# CHAT UI
# ══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about the Quran or Islam…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.status("🔍 Researching your question…", expanded=True) as status:

            # ── Steps 1+2+3 run IN PARALLEL ───────────────────────────────
            st.write("⚡ **Running Steps 1–3 in parallel:**")
            st.write("&nbsp;&nbsp;🧠 Expanding query with LLM  &nbsp;|&nbsp; "
                     "� Loading Quran index  &nbsp;|&nbsp; "
                     "📜 Loading Hadith index")

            terms_box   = [None]
            q_index_box = [None, None]
            h_index_box = [None, None]

            def _expand():
                terms_box[0] = generate_search_terms(user_input)

            def _qidx():
                q_index_box[0], q_index_box[1] = build_quran_index()

            def _hidx():
                h_index_box[0], h_index_box[1] = build_hadith_index()

            with ThreadPoolExecutor(max_workers=3) as pool:
                f1 = pool.submit(_expand)
                f2 = pool.submit(_qidx)
                f3 = pool.submit(_hidx)
                f1.result(); f2.result(); f3.result()

            terms = terms_box[0] or []
            st.write(f"✅ **{len(terms)} search terms** generated: "
                     f"`{', '.join(terms[:7])}{'…' if len(terms)>7 else ''}`")

            # ── BM25 scoring (fast — already indexed) ─────────────────────
            st.write("📊 **BM25Okapi scoring** across full corpus…")
            st.write("&nbsp;&nbsp;&nbsp;&nbsp;↳ Lemmatise + stem query tokens via NLTK")
            st.write("&nbsp;&nbsp;&nbsp;&nbsp;↳ Score all 6,236 verses in one vectorised pass")
            st.write("&nbsp;&nbsp;&nbsp;&nbsp;↳ Score all hadith texts in one vectorised pass")
            st.write("&nbsp;&nbsp;&nbsp;&nbsp;↳ Phrase-match re-ranking applied")

            # Run both retrievals in parallel too
            q_box = [None]
            h_box = [None]
            def _rq(): q_box[0] = retrieve_quran(terms, top_n=15)
            def _rh(): h_box[0] = retrieve_hadith(terms, top_n=15)
            with ThreadPoolExecutor(max_workers=2) as pool:
                pool.submit(_rq).result()
                pool.submit(_rh).result()
            # Note: can't truly parallel here since they share cached index,
            # but sequential BM25 scoring is already <0.5s after index is warm

            q_all = q_box[0] or []
            h_all = h_box[0] or []

            if q_all:
                st.write(f"✅ **Quran:** {len(q_all)} verses ranked — top 8 for answer")
                for i, v in enumerate(q_all[:4], 1):
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;#{i} · "
                             f"*{v['surah_name']}* {v['surah_no']}:{v['ayah_no']}")
                if len(q_all) > 4:
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;…+{len(q_all)-4} more verses")
            else:
                st.write("⚠️ No Quran verses matched.")

            if h_all:
                st.write(f"✅ **Hadith:** {len(h_all)} hadiths ranked — top 6 for answer")
                for i, h in enumerate(h_all[:4], 1):
                    nar = f" · {h['narrator']}" if h["narrator"] else ""
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;#{i} · "
                             f"*{h['book']}* #{h['number']}{nar}")
                if len(h_all) > 4:
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;…+{len(h_all)-4} more hadiths")
            else:
                st.write("⚠️ No Hadiths matched.")

            # ── Step 4: Answer ─────────────────────────────────────────────
            st.write(f"✍️ **Generating answer** — "
                     f"{min(8,len(q_all))} verses + {min(6,len(h_all))} hadiths → Groq 70B…")

            if not q_all and not h_all:
                answer = ("No relevant sources found. Please rephrase your question.")
                status.update(label="❌ No results found", state="error")
            else:
                try:
                    answer = ask_groq(user_input, q_all[:8], h_all[:6])
                    status.update(
                        label=(f"✅ Done · {len(q_all)} verses & {len(h_all)} hadiths searched · "
                               f"{min(8,len(q_all))+min(6,len(h_all))} cited"),
                        state="complete", expanded=False
                    )
                except Exception as e:
                    answer = f"Error: {e}"
                    status.update(label="❌ Error", state="error")

        if q_all:
            with st.expander(f"📖 {len(q_all)} Quran verses (BM25 ranked)"):
                st.text(fmt_q_display(q_all))
        if h_all:
            with st.expander(f"📜 {len(h_all)} Hadiths (BM25 ranked)"):
                st.text(fmt_h_display(h_all))

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<p style="text-align:center;color:#4a7a4a;font-size:0.78rem;margin:0.6rem 0 0.2rem 0;">
  Quran · <a href="https://alquran.cloud">alquran.cloud</a> (Sahih International, 6,236 verses) &nbsp;·&nbsp;
  Hadith · <a href="https://github.com/fawazahmed0/hadith-api">fawazahmed0/hadith-api</a> (Bukhari · Muslim · Abu Dawud · Tirmidhi) &nbsp;·&nbsp;
  Search · BM25Okapi + NLTK &nbsp;·&nbsp;
  LLM · <a href="https://groq.com">Groq</a> Llama 3.3 70B &nbsp;·&nbsp;
  UI · <a href="https://streamlit.io">Streamlit</a>
</p>
<p style="text-align:center;color:#2a4a2a;font-size:0.72rem;margin:0 0 1rem 0;">
  All sources free &amp; open. No user data stored.
</p>
""", unsafe_allow_html=True)
