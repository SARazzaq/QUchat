"""
QUchat — Quran & Hadith Q&A
Search pipeline:
  1. LLM (Groq) analyses question → 35 Islamic search terms
  2. NLTK lemmatisation + PorterStemmer normalises both corpus and query
  3. BM25Okapi (same algorithm as Elasticsearch) ranks all 6,236 verses
     and all hadiths in one pass — no substring guessing
  4. Phrase-match re-ranking boosts multi-word hits
  5. Top results sent to Groq Llama-3.3-70B for cited scholarly answer
"""

import re
import string
import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── NLP imports (lightweight, no model download) ──────────────────────────────
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

# Download required NLTK data (once, cached by Streamlit)
for _pkg in ("punkt", "wordnet", "stopwords", "averaged_perceptron_tagger",
             "punkt_tab", "omw-1.4"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

_stemmer    = PorterStemmer()
_lemmatizer = WordNetLemmatizer()
_stopwords  = set(stopwords.words("english"))

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════
QURAN_FULL_URL = "https://api.alquran.cloud/v1/quran/en.sahih"
HADITH_CDN     = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/{book}.json"
HADITH_BOOKS   = {
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
# NLP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_wordnet_pos(word: str) -> str:
    """Map NLTK POS tag to WordNet POS for accurate lemmatisation."""
    try:
        tag = nltk.pos_tag([word])[0][1][0].upper()
        return {"J": wordnet.ADJ, "V": wordnet.VERB,
                "N": wordnet.NOUN, "R": wordnet.ADV}.get(tag, wordnet.NOUN)
    except Exception:
        return wordnet.NOUN


def normalise(text: str) -> list[str]:
    """
    Full NLP pipeline per token:
    lowercase → tokenise → remove punctuation & stopwords
    → lemmatise (POS-aware) → stem
    Returns list of normalised tokens.
    """
    text = text.lower()
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    result = []
    for t in tokens:
        t = t.strip(string.punctuation)
        if not t or t in _stopwords or not t.isalpha():
            continue
        lemma  = _lemmatizer.lemmatize(t, _get_wordnet_pos(t))
        stem   = _stemmer.stem(lemma)
        result.append(stem)
    return result


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
# STEP 1 — LLM QUERY EXPANSION (Islamic scholar perspective)
# ══════════════════════════════════════════════════════════════════════════════

def generate_search_terms(question: str) -> list[str]:
    """
    Ask Groq to think like an Islamic scholar and produce exhaustive search terms
    covering every theological, linguistic, and contextual dimension.
    """
    prompt = f"""You are an expert Islamic scholar preparing an exhaustive search of the Quran and Hadith.

For the question below, produce a comprehensive list of search terms covering:
1. Core subject (English)
2. Islamic/Arabic transliterated terms (e.g. mawt, barzakh, akhirah, qiyamah, ruh)
3. Synonyms and related English words (die, dying, dead, deceased, perish)
4. Directly connected Islamic concepts (soul, grave, resurrection, judgment day, paradise, hell, angels, trumpet, scales, intercession)
5. Quranic themes linked to this topic
6. Prophets or figures associated (e.g. Israfil, Malak al-Mawt)
7. Events/actions associated (burial, questioning in grave, Day of Judgment)
8. Opposites/contrasts (life, dunya, world)
9. Consequences (reward, punishment, heaven, hellfire)
10. Arabic concept roots that appear in translations (e.g. "mawt" appears as "death", "wafat")

Question: {question}

Output ONLY a Python list of 35-45 lowercase strings — no explanations, no numbering.
["death", "die", "dying", "soul", "ruh", "grave", "barzakh", ...]"""

    try:
        raw   = _groq([{"role": "user", "content": prompt}], max_tokens=600, temp=0.3)
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            terms = eval(match.group())
            if isinstance(terms, list):
                clean = [str(t).strip().lower() for t in terms
                         if t and len(str(t).strip()) > 1]
                return clean[:45]
    except Exception:
        pass
    # Fallback
    return [w.strip("?.,!;:'\"").lower()
            for w in question.split() if len(w) > 2]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — LOAD & INDEX FULL CORPORA (cached 24h)
# Builds BM25 index over normalised tokens — done once, reused for all queries
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def build_quran_index():
    """
    Fetch all 6,236 verses, normalise each with the NLP pipeline,
    build a BM25Okapi index. Returns (verses_list, bm25_index).
    """
    try:
        r = requests.get(QURAN_FULL_URL, timeout=45)
        r.raise_for_status()
        verses = []
        corpus = []   # tokenised docs for BM25
        for s in r.json().get("data", {}).get("surahs", []):
            for a in s.get("ayahs", []):
                text = a["text"]
                verses.append({
                    "surah_no":   s["number"],
                    "surah_name": s["englishName"],
                    "surah_arab": s["name"],
                    "ayah_no":    a["numberInSurah"],
                    "text":       text,
                })
                corpus.append(normalise(text))
        bm25 = BM25Okapi(corpus)
        return verses, bm25
    except Exception:
        return [], None


@st.cache_resource(show_spinner=False)
def build_hadith_index():
    """
    Load all 4 hadith books in parallel, normalise, build BM25 index.
    Returns (hadith_list, bm25_index).
    """
    all_hadiths = []

    def _load(book_name, book_id):
        out = []
        try:
            r = requests.get(HADITH_CDN.format(book=book_id), timeout=45)
            r.raise_for_status()
            for h in r.json().get("hadiths", []):
                text = h.get("text", "")
                if text:
                    out.append({
                        "book":     book_name,
                        "number":   h.get("hadithnumber", ""),
                        "narrator": h.get("by", ""),
                        "text":     text,
                    })
        except Exception:
            pass
        return out

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_load, bn, bi): bn
                   for bn, bi in HADITH_BOOKS.items()}
        for fut in as_completed(futures):
            all_hadiths.extend(fut.result())

    corpus = [normalise(h["text"]) for h in all_hadiths]
    bm25   = BM25Okapi(corpus) if corpus else None
    return all_hadiths, bm25


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — BM25 + PHRASE RE-RANK RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def _phrase_boost(text_lower: str, raw_terms: list[str]) -> float:
    """Extra score for multi-word phrase matches in original text."""
    boost = 0.0
    for t in raw_terms:
        if " " in t and t in text_lower:
            boost += 3.0
    return boost


def retrieve_quran(raw_terms: list[str], top_n: int = 15) -> list[dict]:
    verses, bm25 = build_quran_index()
    if not verses or bm25 is None:
        return []

    # Normalise query terms the same way the index was built
    query_tokens = []
    for t in raw_terms:
        query_tokens.extend(normalise(t))
    # Deduplicate while preserving order
    seen_q: set[str] = set()
    query_tokens = [x for x in query_tokens if not (x in seen_q or seen_q.add(x))]  # type: ignore

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    # Add phrase boost on original text
    for i, v in enumerate(verses):
        scores[i] += _phrase_boost(v["text"].lower(), raw_terms)

    # Get top indices
    top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
    # Filter zero-score results
    top_idx = [i for i in top_idx if scores[i] > 0]

    return [verses[i] for i in top_idx]


def retrieve_hadith(raw_terms: list[str], top_n: int = 15) -> list[dict]:
    hadiths, bm25 = build_hadith_index()
    if not hadiths or bm25 is None:
        return []

    query_tokens = []
    for t in raw_terms:
        query_tokens.extend(normalise(t))
    seen_q: set[str] = set()
    query_tokens = [x for x in query_tokens if not (x in seen_q or seen_q.add(x))]  # type: ignore

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    for i, h in enumerate(hadiths):
        scores[i] += _phrase_boost(h["text"].lower(), raw_terms)

    top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
    top_idx = [i for i in top_idx if scores[i] > 0]

    return [hadiths[i] for i in top_idx]


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fmt_quran_display(verses: list[dict]) -> str:
    return "\n\n".join(
        f"[{v['surah_name']} ({v['surah_arab']}) {v['surah_no']}:{v['ayah_no']}]\n{v['text']}"
        for v in verses
    )


def fmt_hadith_display(hadiths: list[dict]) -> str:
    lines = []
    for h in hadiths:
        nar = f" | {h['narrator']}" if h["narrator"] else ""
        lines.append(f"[{h['book']}, #{h['number']}{nar}]\n{h['text']}")
    return "\n\n".join(lines)


def fmt_quran_llm(verses: list[dict]) -> str:
    return "\n\n".join(
        f'Surah {v["surah_name"]} ({v["surah_arab"]}) — Ch.{v["surah_no"]}, V.{v["ayah_no"]}:\n'
        f'"{v["text"]}"'
        for v in verses
    )


def fmt_hadith_llm(hadiths: list[dict]) -> str:
    lines = []
    for h in hadiths:
        nar  = f" — Narrated by: {h['narrator']}" if h["narrator"] else ""
        text = h["text"][:500] + "…" if len(h["text"]) > 500 else h["text"]
        lines.append(
            f'{h["book"]}, Hadith #{h["number"]}{nar}:\n"{text}"'
        )
    return "\n\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT & ANSWER GENERATION
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a deeply knowledgeable Islamic scholar. Answer using ONLY the Quran verses and Hadiths provided — never fabricate or add external knowledge.

ANSWER FORMAT:

## 📖 Quranic Evidence

For each verse:
**Surah [Name] [Chapter:Verse]**
> *[Quote the verse verbatim in full]*
**Explanation:** 2-4 sentences on exactly what this verse says about the question — its command, ruling, or information, and its significance.

## 📜 Hadith Evidence

For each hadith:
**[Book], Hadith #[N] — Narrated by [Name] (رضي الله عنه/عنها)**
> *[Quote the hadith verbatim in full]*
**Explanation:** 2-4 sentences on the ruling or guidance, its context, and direct relevance to the question.

## ✅ Summary & Islamic Ruling

One comprehensive paragraph giving the complete Islamic answer with inline citations like (Al-Baqarah 2:155) and (Sahih Bukhari #1386). State whether the matter is Fard/Haram/Sunnah/Mubah where applicable.

STRICT RULES:
- Quote every source verbatim before explaining — no exceptions
- ﷺ after Prophet Muhammad every single time
- رضي الله عنه / رضي الله عنها after every Companion name
- Cover ALL provided verses and hadiths — skip none
- Never invent content not in the sources"""


def ask_groq(question: str, q_llm: list[dict], h_llm: list[dict]) -> str:
    q_text = fmt_quran_llm(q_llm)   if q_llm else "None found."
    h_text = fmt_hadith_llm(h_llm)  if h_llm else "None found."

    user_msg = (
        f"Question: {question}\n\n"
        f"=== QURAN (Sahih International) ===\n{q_text}\n\n"
        f"=== HADITH (Bukhari / Muslim / Abu Dawud / Tirmidhi) ===\n{h_text}\n\n"
        f"Write the complete, cited scholarly answer following the mandatory format."
    )
    result = _groq(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user",   "content": user_msg}],
        max_tokens=3000,
        temp=0.1,
    )
    return result or "⚠️ Groq API key not configured. Add GROQ_API_KEY to Streamlit secrets."


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND INDEX WARM-UP
# Trigger index build in a background thread so the UI renders instantly.
# @st.cache_resource ensures it only runs once and all sessions reuse it.
# ══════════════════════════════════════════════════════════════════════════════
import threading

def _warm():
    build_quran_index()
    build_hadith_index()

if "index_warming" not in st.session_state:
    st.session_state["index_warming"] = True
    threading.Thread(target=_warm, daemon=True).start()


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

            # ── Step 1: LLM query expansion ───────────────────────────────
            st.write("🧠 **Step 1/4** — Analysing question from 10 Islamic angles…")
            terms = generate_search_terms(user_input)
            st.write(f"✅ Generated **{len(terms)} search dimensions**: "
                     f"`{', '.join(terms[:8])}{'…' if len(terms) > 8 else ''}`")

            # ── Step 2: Quran BM25 retrieval ──────────────────────────────
            st.write("📖 **Step 2/4** — BM25 ranking across all **6,236 Quran verses** "
                     "(Sahih International)…")
            verses, q_bm25 = build_quran_index()
            if not verses:
                st.write("⏳ Corpus still loading — retrying…")
            st.write("&nbsp;&nbsp;&nbsp;&nbsp;↳ Normalising query: lemmatisation + stemming via NLTK")
            st.write("&nbsp;&nbsp;&nbsp;&nbsp;↳ Scoring every verse with BM25Okapi "
                     "(TF-IDF + document length normalisation)")
            st.write("&nbsp;&nbsp;&nbsp;&nbsp;↳ Phrase-match re-ranking for multi-word terms")
            q_all = retrieve_quran(terms, top_n=15)
            if q_all:
                st.write(f"✅ **{len(q_all)} Quran verses** matched and ranked — "
                         f"top 8 selected for answer")
                for i, v in enumerate(q_all[:5], 1):
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;#{i} → "
                             f"Surah {v['surah_name']} {v['surah_no']}:{v['ayah_no']}")
                if len(q_all) > 5:
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;…and {len(q_all)-5} more")
            else:
                st.write("⚠️ No Quran verses matched.")

            # ── Step 3: Hadith BM25 retrieval ─────────────────────────────
            st.write("📜 **Step 3/4** — BM25 ranking across **4 Hadith collections** "
                     "(Bukhari · Muslim · Abu Dawud · Tirmidhi)…")
            st.write("&nbsp;&nbsp;&nbsp;&nbsp;↳ Searching in parallel across all books")
            st.write("&nbsp;&nbsp;&nbsp;&nbsp;↳ Scoring with BM25Okapi + phrase boost")
            h_all = retrieve_hadith(terms, top_n=15)
            if h_all:
                st.write(f"✅ **{len(h_all)} Hadiths** matched and ranked — "
                         f"top 6 selected for answer")
                for i, h in enumerate(h_all[:4], 1):
                    nar = f" ({h['narrator']})" if h["narrator"] else ""
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;#{i} → "
                             f"{h['book']}, #{h['number']}{nar}")
                if len(h_all) > 4:
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;…and {len(h_all)-4} more")
            else:
                st.write("⚠️ No Hadiths matched.")

            # ── Step 4: Answer generation ──────────────────────────────────
            st.write("✍️ **Step 4/4** — Composing scholarly cited answer with "
                     "Groq Llama 3.3 70B…")
            st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ Sending top **{min(8,len(q_all))} verses** "
                     f"+ **{min(6,len(h_all))} hadiths** as grounded context")

            if not q_all and not h_all:
                answer = ("No relevant verses or Hadiths found. "
                          "Please rephrase or ask about a specific topic.")
                status.update(label="❌ No results found", state="error")
            else:
                try:
                    answer = ask_groq(user_input, q_all[:8], h_all[:6])
                    status.update(
                        label=f"✅ Done — {len(q_all)} verses · {len(h_all)} hadiths searched · "
                              f"{min(8,len(q_all))+min(6,len(h_all))} sources cited",
                        state="complete", expanded=False
                    )
                except Exception as e:
                    answer = f"Error: {e}"
                    status.update(label="❌ Error generating answer", state="error")

        # Show retrieved sources in expanders
        if q_all:
            with st.expander(f"📖 {len(q_all)} Quran verses retrieved (ranked by BM25)"):
                st.text(fmt_quran_display(q_all))
        if h_all:
            with st.expander(f"📜 {len(h_all)} Hadiths retrieved (ranked by BM25)"):
                st.text(fmt_hadith_display(h_all))

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<p style="text-align:center;color:#4a7a4a;font-size:0.78rem;margin:0.6rem 0 0.2rem 0;">
  Quran · <a href="https://alquran.cloud">alquran.cloud</a> (Sahih Int., all 6,236 verses)
  &nbsp;·&nbsp;
  Hadith · <a href="https://github.com/fawazahmed0/hadith-api">fawazahmed0/hadith-api</a>
  (Bukhari · Muslim · Abu Dawud · Tirmidhi)
  &nbsp;·&nbsp;
  Search · BM25Okapi + NLTK lemmatisation/stemming
  &nbsp;·&nbsp;
  LLM · <a href="https://groq.com">Groq</a> Llama 3.3 70B
  &nbsp;·&nbsp;
  UI · <a href="https://streamlit.io">Streamlit</a>
</p>
<p style="text-align:center;color:#2a4a2a;font-size:0.72rem;margin:0 0 1rem 0;">
  All sources free &amp; open. No user data stored.
</p>
""", unsafe_allow_html=True)
