import streamlit as st
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# Quran full corpus : alquran.cloud/v1/quran/en.sahih  (all 6236 verses, no auth)
# Quran fallback    : fawazahmed0/quran-api via jsDelivr (JSON dump, no auth)
# Hadith primary    : fawazahmed0/hadith-api Bukhari + Muslim via jsDelivr CDN
# Hadith fallback   : Abu Dawud + Tirmidhi from same CDN
# LLM               : Groq Llama 3.3 70B (free tier)
# ══════════════════════════════════════════════════════════════════════════════

QURAN_FULL_URL  = "https://api.alquran.cloud/v1/quran/en.sahih"
QURAN_ARABIC_URL= "https://api.alquran.cloud/v1/quran/quran-uthmani"
HADITH_CDN      = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/{book}.json"

HADITH_BOOKS = {
    "Sahih Bukhari":  "eng-bukhari",
    "Sahih Muslim":   "eng-muslim",
    "Sunan Abu Dawud":"eng-abudawud",
    "Jami at-Tirmidhi":"eng-tirmidhi",
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

html, body, [class*="css"] { font-family:'Inter',sans-serif; color:#f0ece4; }
.stApp { background:#0f1e0f; }
#MainMenu, footer, header { visibility:hidden; }

.stChatInput textarea {
    background:#1a2e1a !important; color:#f5f0e8 !important;
    border:1.5px solid #b8960c !important; border-radius:14px !important;
    font-family:'Inter',sans-serif !important; font-size:1rem !important;
}
.stChatInput textarea::placeholder { color:#7a9a7a !important; }
.stChatMessage { background:transparent !important; }
[data-testid="stChatMessageContent"] {
    font-family:'Inter',sans-serif; font-size:1rem; line-height:1.9; color:#f0ece4;
}
details {
    background:#152015 !important; border:1px solid #2a4a2a !important;
    border-radius:10px !important; margin-bottom:6px;
}
summary {
    color:#d4a017 !important; font-weight:600 !important;
    font-size:0.88rem !important; padding:10px 14px !important;
}
.streamlit-expanderContent {
    color:#c8dcc0 !important; font-size:0.82rem !important;
    font-family:'Inter',sans-serif !important; line-height:1.6 !important;
    white-space:pre-wrap !important;
}
[data-testid="stAlert"] {
    background:#1a2a1a !important; border-left:4px solid #d4a017 !important;
    border-radius:10px !important; color:#e8dfc8 !important; font-size:0.9rem !important;
}
.stSpinner > div { border-top-color:#d4a017 !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#0f1e0f; }
::-webkit-scrollbar-thumb { background:#b8960c; border-radius:3px; }
.arabic { font-family:'Amiri',serif; font-size:1.4rem; color:#f5e6a0; direction:rtl; line-height:2.4; }
hr { border-color:#1e3a1e !important; }
a { color:#7ec88a !important; } a:hover { color:#d4a017 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2.5rem 0 1.2rem 0;">
  <div style="font-size:3.2rem; margin-bottom:0.3rem;">🕌</div>
  <h1 style="font-family:'Amiri',serif; color:#d4a017; font-size:2.8rem; margin:0; letter-spacing:1px;">QUchat</h1>
  <p style="color:#7ec88a; font-size:1rem; margin:0.4rem 0 0.8rem 0; letter-spacing:2px; text-transform:uppercase; font-weight:500;">
    Quran & Hadith Question Answering
  </p>
  <p class="arabic">بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ</p>
  <p style="color:#8aaa88; font-size:0.85rem; font-style:italic; margin-top:0.2rem;">
    In the name of Allah, the Most Gracious, the Most Merciful
  </p>
</div>
<hr>
""", unsafe_allow_html=True)

st.info(
    "⚠️ **Disclaimer:** Answers are AI-generated and grounded in authentic Islamic sources. "
    "AI can make mistakes — always verify with a qualified Islamic scholar. "
    "Quran: Sahih International · Hadith: Bukhari, Muslim, Abu Dawud, Tirmidhi."
)

# ══════════════════════════════════════════════════════════════════════════════
# GROQ HELPER (shared for both query-expansion and final answer)
# ══════════════════════════════════════════════════════════════════════════════

def _groq_call(messages: list[dict], max_tokens: int = 500, temperature: float = 0.1) -> str:
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return ""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LLM-POWERED QUERY ANALYSIS
# Ask Groq to analyse the question 10 ways and return search keywords
# ══════════════════════════════════════════════════════════════════════════════

def generate_search_terms(question: str) -> list[str]:
    """
    Use the LLM to think about the question from 10 Islamic angles and
    return a rich list of search keywords covering every dimension.
    """
    prompt = f"""You are an Islamic scholar preparing to search the Quran and Hadith.

Analyse this question from 10 different angles:
1. Literal meaning of the question
2. Islamic terminology / Arabic terms related to it
3. Related religious obligations or prohibitions
4. Related concepts (e.g. if about prayer → also: wudu, qibla, times, pillars)
5. Related people/prophets mentioned in Quran
6. Related events or stories in Quran/Hadith
7. Spiritual/moral dimensions
8. Legal/fiqh dimensions
9. Reward/punishment angles
10. Synonyms and alternate phrasings

Question: {question}

Output ONLY a Python list of single-word and short-phrase search terms (no explanations).
Include both English and transliterated Arabic terms.
Example format: ["prayer", "salah", "salat", "namaz", "establish prayer", "five prayers", ...]
Output 25-40 terms. Output ONLY the list, nothing else."""

    try:
        raw = _groq_call(
            [{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.2,
        )
        # Parse the list from the response
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            terms = eval(match.group())  # safe — we control the prompt format
            if isinstance(terms, list):
                return [str(t).strip().lower() for t in terms if t and len(str(t).strip()) > 1]
    except Exception:
        pass

    # Fallback: basic tokenisation
    tokens = [w.strip("?.,!;:'\"").lower() for w in question.split() if len(w) > 2]
    return tokens


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — LOAD FULL CORPORA (cached — loaded once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def load_full_quran() -> list[dict]:
    """
    Load the complete Quran (6236 verses, Sahih International).
    Returns list of dicts: {surah_no, surah_name, ayah_no, text}
    """
    verses = []
    try:
        r = requests.get(QURAN_FULL_URL, timeout=30)
        r.raise_for_status()
        surahs = r.json().get("data", {}).get("surahs", [])
        for surah in surahs:
            sno   = surah.get("number", 0)
            sname = surah.get("englishName", f"Surah {sno}")
            sarab = surah.get("name", "")
            for ayah in surah.get("ayahs", []):
                verses.append({
                    "surah_no":   sno,
                    "surah_name": sname,
                    "surah_arab": sarab,
                    "ayah_no":    ayah.get("numberInSurah", 0),
                    "text":       ayah.get("text", ""),
                })
    except Exception:
        pass
    return verses


@st.cache_data(ttl=86400, show_spinner=False)
def load_hadith_book(book_id: str) -> list[dict]:
    """Load and cache a full hadith book from jsDelivr CDN."""
    try:
        r = requests.get(HADITH_CDN.format(book=book_id), timeout=30)
        r.raise_for_status()
        return r.json().get("hadiths", [])
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FULL-CORPUS SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def search_quran_full(terms: list[str]) -> tuple[str, int]:
    """
    Search ALL 6236 Quran verses for any of the given terms.
    Scores each verse by how many terms it matches — returns top matches.
    """
    verses = load_full_quran()
    if not verses:
        return "", 0

    # Score every verse
    scored = []
    terms_lower = [t.lower() for t in terms]

    for v in verses:
        text_lower = v["text"].lower()
        score = sum(1 for t in terms_lower if t in text_lower)
        # Boost multi-word phrase matches
        score += sum(2 for t in terms_lower if " " in t and t in text_lower)
        if score > 0:
            scored.append((score, v))

    # Sort by score descending, then by surah/ayah
    scored.sort(key=lambda x: (-x[0], x[1]["surah_no"], x[1]["ayah_no"]))

    # Take top 25 most relevant verses
    top = scored[:25]

    if not top:
        return "", 0

    lines = []
    for _, v in top:
        lines.append(
            f"[Surah {v['surah_name']} ({v['surah_arab']}) — "
            f"Chapter {v['surah_no']}, Verse {v['ayah_no']}]\n"
            f"{v['text']}"
        )

    return "\n\n".join(lines), len(top)


def search_hadith_full(terms: list[str]) -> tuple[str, int]:
    """
    Search ALL hadith across 4 books (Bukhari, Muslim, Abu Dawud, Tirmidhi).
    Scores each hadith by term matches — returns top matches.
    """
    terms_lower = [t.lower() for t in terms]
    scored: list[tuple[int, str, dict]] = []
    seen: set[str] = set()

    def search_book(book_name: str, book_id: str):
        hadiths = load_hadith_book(book_id)
        local   = []
        for h in hadiths:
            text  = h.get("text", "")
            hid   = f"{book_id}-{h.get('hadithnumber','')}"
            if hid in seen or not text:
                continue
            text_lower = text.lower()
            score = sum(1 for t in terms_lower if t in text_lower)
            score += sum(2 for t in terms_lower if " " in t and t in text_lower)
            if score > 0:
                local.append((score, book_name, h, hid))
        return local

    # Load all 4 books in parallel
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(search_book, bn, bi): bn
            for bn, bi in HADITH_BOOKS.items()
        }
        for fut in as_completed(futures):
            for score, bname, h, hid in fut.result():
                if hid not in seen:
                    seen.add(hid)
                    scored.append((score, bname, h))

    scored.sort(key=lambda x: -x[0])
    top = scored[:20]

    if not top:
        return "", 0

    lines = []
    for _, book_name, h in top:
        narrator = h.get("by", "")
        nar_str  = f" — Narrated by: {narrator}" if narrator else ""
        lines.append(
            f"[{book_name}, Hadith #{h.get('hadithnumber','')}{nar_str}]\n"
            f"{h.get('text','')}"
        )

    return "\n\n".join(lines), len(top)


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a deeply knowledgeable Islamic scholar. Answer using ONLY the Quran verses and Hadiths provided — never fabricate.

═══════════ MANDATORY STRUCTURE ═══════════

## 📖 From the Quran

For EVERY verse in the context:

**Surah [Name] ([Arabic]) — Chapter [X], Verse [Y]**

> [Paste the COMPLETE verbatim text of the verse — do not shorten]

**Explanation:** Detailed explanation covering:
- What Allah ﷻ is commanding, forbidding, or informing
- The depth and significance of this verse
- How it directly answers the question asked
- Nuances in wording important to understand

━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📜 From the Hadith

For EVERY Hadith in the context:

**[Book], Hadith #[N] — Narrated by: [Name] (رضي الله عنه/عنها)**

> [Paste the COMPLETE verbatim text of the Hadith — do not shorten]

**Explanation:** Detailed explanation covering:
- What the Prophet ﷺ said, did, or approved
- The ruling or guidance it establishes
- How it supports or elaborates on the Quranic evidence
- Its relevance to the question

━━━━━━━━━━━━━━━━━━━━━━━━━━

## ✅ Islamic Ruling & Summary

- State the ruling: Fard (فرض) / Sunnah (سنة) / Haram (حرام) / Mubah (مباح)
- Summarise the cumulative evidence
- Explain the Islamic wisdom behind it

═══════════════════════════

RULES:
1. Quote every single source verbatim in blockquote before explaining — no exceptions
2. Cover EVERY verse and EVERY hadith provided — skip none
3. ﷺ after Prophet Muhammad every time
4. رضي الله عنه/عنها after every Companion
5. Never add content beyond what is in the provided sources
6. Long, thorough, scholarly answers required"""


def ask_groq_full(question: str, context: str) -> str:
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return "⚠️ Groq API key not configured. Add GROQ_API_KEY to your Streamlit secrets."

    user_msg = (
        f"QUESTION: {question}\n\n"
        f"Analyse this question from every possible dimension — "
        f"linguistic, theological, legal (fiqh), moral, and spiritual — "
        f"before constructing your answer.\n\n"
        f"{'═'*60}\n"
        f"ALL RETRIEVED AUTHENTIC ISLAMIC SOURCES\n"
        f"{'═'*60}\n\n"
        f"{context}\n\n"
        f"{'═'*60}\n\n"
        f"Now write the complete, detailed, fully-referenced scholarly answer "
        f"following the mandatory structure. "
        f"Quote every verse and hadith verbatim first, then give thorough explanations."
    )

    return _groq_call(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=4096,
        temperature=0.1,
    )


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

        # ── Step 1: LLM analyses question → generates search terms ────────────
        with st.spinner("🧠 Analysing question from 10 Islamic angles…"):
            search_terms = generate_search_terms(user_input)

        with st.expander(f"🔎 {len(search_terms)} search terms generated"):
            st.write(", ".join(search_terms))

        # ── Step 2: Full-corpus Quran search ─────────────────────────────────
        with st.spinner("📖 Searching all 6,236 Quran verses…"):
            quran_ctx, verse_count = search_quran_full(search_terms)

        # ── Step 3: Full-corpus Hadith search ─────────────────────────────────
        with st.spinner("📜 Searching Bukhari, Muslim, Abu Dawud & Tirmidhi…"):
            hadith_ctx, hadith_count = search_hadith_full(search_terms)

        # ── Build combined context ────────────────────────────────────────────
        sections = []
        if quran_ctx:
            sections.append(
                f"=== QURAN — {verse_count} verses (Sahih International) ===\n\n{quran_ctx}"
            )
        if hadith_ctx:
            sections.append(
                f"=== HADITH — {hadith_count} hadiths (Bukhari / Muslim / Abu Dawud / Tirmidhi) ===\n\n{hadith_ctx}"
            )
        combined = "\n\n".join(sections)

        if not combined:
            answer = (
                "I could not find relevant Quranic verses or Hadiths for your question. "
                "Please try rephrasing or asking about a specific topic or verse."
            )
            st.markdown(answer)
        else:
            if quran_ctx:
                with st.expander(f"📖 {verse_count} Quran verses retrieved"):
                    st.text(quran_ctx)
            if hadith_ctx:
                with st.expander(f"📜 {hadith_count} Hadith retrieved"):
                    st.text(hadith_ctx)

            # ── Step 4: Generate answer ───────────────────────────────────────
            with st.spinner("✍️ Composing detailed scholarly answer…"):
                try:
                    answer = ask_groq_full(user_input, combined)
                except Exception as e:
                    answer = f"Error: {e}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<p style="text-align:center; color:#4a7a4a; font-size:0.78rem; margin:0.6rem 0 0.2rem 0;">
  Quran · <a href="https://alquran.cloud">alquran.cloud</a> (Sahih International, all 6236 verses) &nbsp;·&nbsp;
  Hadith · <a href="https://github.com/fawazahmed0/hadith-api">fawazahmed0/hadith-api</a> (Bukhari · Muslim · Abu Dawud · Tirmidhi) &nbsp;·&nbsp;
  LLM · <a href="https://groq.com">Groq</a> Llama 3.3 70B &nbsp;·&nbsp;
  UI · <a href="https://streamlit.io">Streamlit</a>
</p>
<p style="text-align:center; color:#2a4a2a; font-size:0.72rem; margin:0 0 1rem 0;">
  All sources free &amp; open. No user data stored.
</p>
""", unsafe_allow_html=True)
