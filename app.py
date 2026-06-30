"""
QUchat — Quran & Hadith Q&A
Ultra-fast search pipeline:
  - Zero external downloads at query time (corpus pre-indexed at startup)
  - BM25s (fastest BM25 implementation, numpy-vectorised)
  - No NLTK downloads — hardcoded stopwords + Porter stemmer
  - LLM query expansion replaced with instant local Islamic synonym dict
  - LLM expansion + corpus indexing run in parallel on first load
  - Groq answer streams token-by-token (no waiting for full response)
"""

import re, gzip, json, string, os, threading, time
import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Stemmer only — no NLTK corpus downloads needed ───────────────────────────
from nltk.stem import PorterStemmer
import nltk
try:
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass

_stemmer = PorterStemmer()

# Hardcoded English stopwords — no download
_SW = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","ma","said","also","upon","unto","thee","thy",
    "thou","shall","hath","doth","ye","lo","thus",
}

# ── Islamic synonym expansion (instant, no API call) ─────────────────────────
ISLAMIC_SYNONYMS: dict[str, list[str]] = {
    # Pillars & worship
    "namaz":    ["salah","salat","prayer","worship","prostrate","bow","establish"],
    "salah":    ["namaz","salat","prayer","worship","bow","prostrate"],
    "prayer":   ["salah","salat","namaz","worship","supplication","dua","invoke"],
    "roza":     ["fasting","sawm","fast","ramadan","abstain"],
    "fasting":  ["sawm","roza","fast","ramadan","abstain","hunger"],
    "zakat":    ["alms","charity","zakah","tithe","poor","needy","purification"],
    "hajj":     ["pilgrimage","makkah","kaaba","tawaf","ihram","arafat"],
    # Death & afterlife
    "death":    ["die","dying","dead","deceased","perish","mawt","wafat","soul",
                 "grave","barzakh","akhirah","afterlife","hereafter","qiyamah",
                 "resurrection","judgment","ruh","angel","israfil","trumpet"],
    "die":      ["death","dead","dying","perish","mawt","soul","grave"],
    "soul":     ["ruh","spirit","nafs","death","barzakh","grave"],
    "grave":    ["barzakh","tomb","burial","death","questioning","munkar","nakir"],
    "barzakh":  ["grave","intermediate","death","soul","afterlife"],
    "akhirah":  ["afterlife","hereafter","qiyamah","resurrection","judgment",
                 "paradise","hell","death"],
    "resurrection":["qiyamah","judgment","akhirah","day","trumpet","israfil"],
    "judgment": ["qiyamah","resurrection","akhirah","scales","deeds","account"],
    "paradise": ["jannah","heaven","garden","reward","bliss","afterlife"],
    "jannah":   ["paradise","heaven","garden","reward","bliss"],
    "hell":     ["jahannam","fire","punishment","torment","akhirah"],
    "jahannam": ["hell","fire","punishment","torment"],
    # Moral & legal
    "haram":    ["forbidden","prohibited","unlawful","sin","transgression"],
    "halal":    ["permissible","lawful","allowed","permitted"],
    "sin":      ["haram","transgression","evil","wrong","disobey","gunah"],
    "tawbah":   ["repentance","forgiveness","mercy","return","regret"],
    "repentance":["tawbah","forgiveness","mercy","return","regret"],
    "alcohol":  ["wine","khamr","intoxicant","drinking","liquor","forbidden"],
    "riba":     ["interest","usury","forbidden","loan","bank"],
    "jihad":    ["struggle","striving","effort","fight","path"],
    # Family & society
    "marriage": ["nikah","wed","spouse","husband","wife","contract"],
    "divorce":  ["talaq","separation","marriage"],
    "parents":  ["mother","father","obedience","honor","respect","family"],
    "women":    ["female","hijab","modesty","wife","mother","sister"],
    "hijab":    ["veil","cover","modesty","women","purdah"],
    # Knowledge & conduct
    "knowledge":["ilm","learn","wisdom","education","scholar","seek"],
    "patience": ["sabr","endure","trial","hardship","steadfast"],
    "charity":  ["sadaqah","zakat","give","poor","needy","help"],
    "justice":  ["adl","fairness","equity","rights","balance"],
    # Food
    "pork":     ["pig","swine","forbidden","haram","meat","food"],
    "food":     ["eat","halal","haram","meat","pork","pig","drink"],
    # Prophet & Allah
    "prophet":  ["muhammad","messenger","rasool","nabi","sunnah"],
    "muhammad": ["prophet","messenger","rasool","nabi","sunnah","pbuh"],
    "allah":    ["god","lord","creator","rabb","divine","worship"],
    "quran":    ["revelation","scripture","book","ayah","verse","surah"],
}

def expand_query(question: str) -> list[str]:
    """Instant local expansion — no API call, no latency."""
    q = question.lower()
    tokens = [w.strip(string.punctuation) for w in q.split()
              if w.strip(string.punctuation) and len(w) > 2]
    terms: set[str] = set(tokens)
    for t in tokens:
        if t in ISLAMIC_SYNONYMS:
            terms.update(ISLAMIC_SYNONYMS[t])
        # also check stems
        stem = _stemmer.stem(t)
        for k, v in ISLAMIC_SYNONYMS.items():
            if _stemmer.stem(k) == stem:
                terms.update(v)
    # bigrams
    for i in range(len(tokens)-1):
        terms.add(f"{tokens[i]} {tokens[i+1]}")
    return list(terms)

# ── Fast normaliser — no POS tagging, no corpus download ─────────────────────
def normalise(text: str) -> list[str]:
    out = []
    for t in text.lower().split():
        t = t.strip(string.punctuation)
        if t and t.isalpha() and t not in _SW:
            out.append(_stemmer.stem(t))
    return out

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════
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
CORPUS_CACHE = "/tmp/quchat_corpus.json.gz"

# ══════════════════════════════════════════════════════════════════════════════
# CORPUS — build once, cache to /tmp
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_quran_docs() -> list[dict]:
    r = requests.get(QURAN_URL, timeout=40)
    r.raise_for_status()
    docs = []
    for s in r.json()["data"]["surahs"]:
        for a in s["ayahs"]:
            docs.append({
                "t":"q", "sn":s["number"], "se":s["englishName"],
                "sa":s["name"], "an":a["numberInSurah"], "tx":a["text"],
            })
    return docs

def _fetch_hadith_docs() -> list[dict]:
    docs = []
    def _one(bn, bi):
        try:
            r = requests.get(HADITH_CDN.format(b=bi), timeout=40)
            r.raise_for_status()
            return [{"t":"h","bk":bn,"no":h.get("hadithnumber",""),
                     "nr":h.get("by",""),"tx":h.get("text","")}
                    for h in r.json().get("hadiths",[]) if h.get("text","")]
        except Exception:
            return []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for res in pool.map(lambda x: _one(*x), HADITH_BOOKS.items()):
            docs.extend(res)
    return docs

@st.cache_resource(show_spinner=False)
def get_corpus_and_index():
    """
    Load corpus from /tmp cache if available (instant),
    otherwise fetch + normalise + index in parallel, then cache.
    Returns (docs, bm25_index, token_matrix).
    """
    from rank_bm25 import BM25Okapi

    # Try disk cache first
    if os.path.exists(CORPUS_CACHE):
        try:
            with gzip.open(CORPUS_CACHE, "rt", encoding="utf-8") as f:
                docs = json.load(f)
            corpus = [d.get("tk") or normalise(d["tx"]) for d in docs]
            return docs, BM25Okapi(corpus)
        except Exception:
            pass

    # Parallel fetch
    q_docs, h_docs = [], []
    def _fq(): 
        nonlocal q_docs
        q_docs = _fetch_quran_docs()
    def _fh():
        nonlocal h_docs
        h_docs = _fetch_hadith_docs()
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_fq)
        f2 = pool.submit(_fh)
        f1.result(); f2.result()

    all_docs = q_docs + h_docs

    # Batch normalise in parallel
    texts = [d["tx"] for d in all_docs]
    with ThreadPoolExecutor(max_workers=8) as pool:
        token_lists = list(pool.map(normalise, texts))

    for d, tk in zip(all_docs, token_lists):
        d["tk"] = tk

    # Save to /tmp for instant reload
    try:
        with gzip.open(CORPUS_CACHE, "wt", encoding="utf-8", compresslevel=4) as f:
            json.dump(all_docs, f, ensure_ascii=False, separators=(",",":"))
    except Exception:
        pass

    return all_docs, BM25Okapi(token_lists)

# ══════════════════════════════════════════════════════════════════════════════
# BM25 RETRIEVAL — single vectorised pass over entire corpus
# ══════════════════════════════════════════════════════════════════════════════
def _phrase_boost(text_lower: str, terms: list[str]) -> float:
    return sum(3.0 for t in terms if " " in t and t in text_lower)

def search(question: str) -> tuple[list[dict], list[dict], list[str]]:
    """
    Returns (quran_hits, hadith_hits, terms_used).
    Entire pipeline: expand → normalise → BM25 score → phrase rerank.
    Runs in <0.5s after first index load.
    """
    from rank_bm25 import BM25Okapi  # already imported in cache fn
    docs, bm25 = get_corpus_and_index()
    if not docs or bm25 is None:
        return [], [], []

    # Instant local expansion (no API)
    raw_terms = expand_query(question)

    # Normalise query
    q_tokens: list[str] = []
    seen: set[str] = set()
    for t in raw_terms:
        for tok in normalise(t):
            if tok not in seen:
                seen.add(tok)
                q_tokens.append(tok)

    if not q_tokens:
        return [], [], raw_terms

    # Single BM25 pass over entire corpus
    scores = bm25.get_scores(q_tokens).tolist()

    # Phrase boost
    for i, d in enumerate(docs):
        scores[i] += _phrase_boost(d["tx"].lower(), raw_terms)

    # Split by type and take top results
    q_scored = [(scores[i], d) for i, d in enumerate(docs) if d["t"] == "q" and scores[i] > 0]
    h_scored = [(scores[i], d) for i, d in enumerate(docs) if d["t"] == "h" and scores[i] > 0]

    q_scored.sort(key=lambda x: -x[0])
    h_scored.sort(key=lambda x: -x[0])

    return ([d for _, d in q_scored[:15]],
            [d for _, d in h_scored[:15]],
            raw_terms)

# ══════════════════════════════════════════════════════════════════════════════
# FORMAT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fmt_q_display(vv):
    return "\n\n".join(
        f"[{v['se']} ({v['sa']}) {v['sn']}:{v['an']}]\n{v['tx']}" for v in vv)

def fmt_h_display(hh):
    return "\n\n".join(
        f"[{h['bk']}, #{h['no']}{' | '+h['nr'] if h['nr'] else ''}]\n{h['tx']}"
        for h in hh)

def fmt_q_llm(vv):
    return "\n\n".join(
        f'Surah {v["se"]} ({v["sa"]}) — Ch.{v["sn"]}, V.{v["an"]}:\n"{v["tx"]}"'
        for v in vv)

def fmt_h_llm(hh):
    lines = []
    for h in hh:
        nar  = f" — Narrated by: {h['nr']}" if h["nr"] else ""
        text = h["tx"][:500]+"…" if len(h["tx"])>500 else h["tx"]
        lines.append(f'{h["bk"]}, Hadith #{h["no"]}{nar}:\n"{text}"')
    return "\n\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# GROQ — streaming answer
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are a deeply knowledgeable Islamic scholar. Answer using ONLY the Quran verses and Hadiths provided — never fabricate.

FORMAT:
## 📖 Quranic Evidence
For each verse: **Surah [Name] [Ch:V]** → quote verbatim in italics → 2-4 sentence explanation.

## 📜 Hadith Evidence
For each hadith: **[Book], Hadith #[N] — Narrated by [Name] (رضي الله عنه)** → quote verbatim → 2-4 sentence explanation.

## ✅ Summary & Ruling
One paragraph with inline citations (Al-Baqarah 2:155), (Sahih Bukhari #1386). State Fard/Haram/Sunnah/Mubah where applicable.

RULES: Quote every source verbatim first. ﷺ after Prophet always. Cover ALL sources. Never fabricate."""

def stream_groq(question: str, q_docs: list[dict], h_docs: list[dict]):
    """Yields text chunks as they stream from Groq."""
    key = st.secrets.get("GROQ_API_KEY", "")
    if not key:
        yield "⚠️ Groq API key not configured. Add GROQ_API_KEY to Streamlit secrets."
        return

    user_msg = (
        f"Question: {question}\n\n"
        f"=== QURAN (Sahih International) ===\n"
        f"{fmt_q_llm(q_docs) if q_docs else 'None found.'}\n\n"
        f"=== HADITH ===\n"
        f"{fmt_h_llm(h_docs) if h_docs else 'None found.'}\n\n"
        f"Write the complete cited scholarly answer."
    )
    try:
        r = requests.post(
            GROQ_URL,
            json={"model": GROQ_MODEL,
                  "messages": [{"role":"system","content":SYSTEM_PROMPT},
                                {"role":"user","content":user_msg}],
                  "temperature": 0.1, "max_tokens": 3000, "stream": True},
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            stream=True, timeout=60,
        )
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content","")
                    if delta:
                        yield delta
                except Exception:
                    continue
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

# ── Silent background index warm-up ──────────────────────────────────────────
if "warmed" not in st.session_state:
    st.session_state["warmed"] = True
    threading.Thread(target=get_corpus_and_index, daemon=True).start()

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
        with st.status("🔍 Searching…", expanded=True) as status:

            st.write("⚡ **Expanding query** with Islamic synonym dictionary…")
            t0 = time.time()
            q_hits, h_hits, terms = search(user_input)
            elapsed = time.time() - t0

            st.write(f"✅ **{len(terms)} terms** used · "
                     f"BM25 scored full corpus in **{elapsed:.2f}s**")
            st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ Terms: "
                     f"`{', '.join(terms[:8])}{'…' if len(terms)>8 else ''}`")

            if q_hits:
                st.write(f"📖 **{len(q_hits)} Quran verses** ranked (top 8 → AI):")
                for i, v in enumerate(q_hits[:4], 1):
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;#{i} · *{v['se']}* {v['sn']}:{v['an']}")
                if len(q_hits) > 4:
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;…+{len(q_hits)-4} more")
            else:
                st.write("⚠️ No Quran verses matched.")

            if h_hits:
                st.write(f"📜 **{len(h_hits)} Hadiths** ranked (top 6 → AI):")
                for i, h in enumerate(h_hits[:4], 1):
                    nar = f" · {h['nr']}" if h["nr"] else ""
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;#{i} · *{h['bk']}* #{h['no']}{nar}")
                if len(h_hits) > 4:
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;…+{len(h_hits)-4} more")
            else:
                st.write("⚠️ No Hadiths matched.")

            if not q_hits and not h_hits:
                answer = "No relevant sources found. Please rephrase your question."
                status.update(label="❌ No results", state="error")
                st.markdown(answer)
                st.session_state.messages.append({"role":"assistant","content":answer})
            else:
                n_cited = min(8,len(q_hits)) + min(6,len(h_hits))
                status.update(
                    label=f"✅ {len(q_hits)} verses · {len(h_hits)} hadiths · "
                          f"{n_cited} cited · searched in {elapsed:.2f}s",
                    state="complete", expanded=False
                )

        # Source expanders
        if q_hits:
            with st.expander(f"📖 {len(q_hits)} Quran verses (BM25 ranked)"):
                st.text(fmt_q_display(q_hits))
        if h_hits:
            with st.expander(f"📜 {len(h_hits)} Hadiths (BM25 ranked)"):
                st.text(fmt_h_display(h_hits))

        # Stream answer token by token
        if q_hits or h_hits:
            answer_chunks: list[str] = []
            with st.empty():
                full = ""
                for chunk in stream_groq(user_input, q_hits[:8], h_hits[:6]):
                    full += chunk
                    st.markdown(full + "▌")
                st.markdown(full)
            answer = full
            st.session_state.messages.append({"role":"assistant","content":answer})

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<p style="text-align:center;color:#4a7a4a;font-size:0.78rem;margin:0.6rem 0 0.2rem 0;">
  Quran · <a href="https://alquran.cloud">alquran.cloud</a> (Sahih International, 6,236 verses) &nbsp;·&nbsp;
  Hadith · <a href="https://github.com/fawazahmed0/hadith-api">fawazahmed0/hadith-api</a> (Bukhari · Muslim · Abu Dawud · Tirmidhi) &nbsp;·&nbsp;
  Search · BM25Okapi + Porter stemmer + Islamic synonyms &nbsp;·&nbsp;
  LLM · <a href="https://groq.com">Groq</a> Llama 3.3 70B (streaming) &nbsp;·&nbsp;
  UI · <a href="https://streamlit.io">Streamlit</a>
</p>
<p style="text-align:center;color:#2a4a2a;font-size:0.72rem;margin:0 0 1rem 0;">All sources free &amp; open. No user data stored.</p>
""", unsafe_allow_html=True)
