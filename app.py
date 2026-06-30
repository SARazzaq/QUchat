import streamlit as st
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS  — all free, no auth
# ══════════════════════════════════════════════════════════════════════════════
QURAN_FULL_URL = "https://api.alquran.cloud/v1/quran/en.sahih"
HADITH_CDN     = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/{book}.json"
HADITH_BOOKS   = {
    "Sahih Bukhari":   "eng-bukhari",
    "Sahih Muslim":    "eng-muslim",
    "Sunan Abu Dawud": "eng-abudawud",
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
html,body,[class*="css"]{font-family:'Inter',sans-serif;color:#f0ece4;}
.stApp{background:#0f1e0f;}
#MainMenu,footer,header{visibility:hidden;}
.stChatInput textarea{background:#1a2e1a!important;color:#f5f0e8!important;border:1.5px solid #b8960c!important;border-radius:14px!important;font-family:'Inter',sans-serif!important;font-size:1rem!important;}
.stChatInput textarea::placeholder{color:#7a9a7a!important;}
.stChatMessage{background:transparent!important;}
[data-testid="stChatMessageContent"]{font-family:'Inter',sans-serif;font-size:1rem;line-height:1.9;color:#f0ece4;}
details{background:#152015!important;border:1px solid #2a4a2a!important;border-radius:10px!important;margin-bottom:6px;}
summary{color:#d4a017!important;font-weight:600!important;font-size:0.88rem!important;padding:10px 14px!important;}
.streamlit-expanderContent{color:#c8dcc0!important;font-size:0.82rem!important;font-family:'Inter',sans-serif!important;line-height:1.6!important;white-space:pre-wrap!important;}
[data-testid="stAlert"]{background:#1a2a1a!important;border-left:4px solid #d4a017!important;border-radius:10px!important;color:#e8dfc8!important;font-size:0.9rem!important;}
.stSpinner>div{border-top-color:#d4a017!important;}
::-webkit-scrollbar{width:5px;}::-webkit-scrollbar-track{background:#0f1e0f;}::-webkit-scrollbar-thumb{background:#b8960c;border-radius:3px;}
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
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=40,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LLM QUERY ANALYSIS
# Generate rich, multi-dimensional search terms from the question
# ══════════════════════════════════════════════════════════════════════════════
def generate_search_terms(question: str) -> list[str]:
    prompt = f"""You are an expert Islamic scholar preparing to do a deep search of the Quran and Hadith.

Given this question, think deeply and produce an exhaustive list of search terms that covers:
1. Core subject (literal terms)
2. Arabic/Islamic terms (transliterated): e.g. mawt, barzakh, akhirah, qiyamah
3. Synonyms and related English words
4. Concepts directly connected: e.g. "death" → soul, grave, resurrection, judgement, paradise, hell, barzakh, ruh, angel of death, trumpet, scales
5. Quranic themes associated with this topic
6. People/Prophets associated (e.g. Israfil for trumpet)
7. Actions/events associated
8. Opposite concepts (e.g. life, dunya)
9. Consequences and outcomes
10. Any Hadith-specific terms

Question: {question}

Return ONLY a Python list of 30-40 lowercase strings. No explanations. Example:
["death", "die", "dying", "dead", "soul", "ruh", "grave", ...]"""

    try:
        raw   = _groq([{"role": "user", "content": prompt}], max_tokens=500, temp=0.3)
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            terms = eval(match.group())
            if isinstance(terms, list):
                return [str(t).strip().lower() for t in terms if t and len(str(t).strip()) > 1]
    except Exception:
        pass
    return [w.strip("?.,!;:'\"").lower() for w in question.split() if len(w) > 2]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — LOAD FULL CORPORA (cached 24h)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=86400, show_spinner=False)
def load_full_quran() -> list[dict]:
    try:
        r = requests.get(QURAN_FULL_URL, timeout=40)
        r.raise_for_status()
        out = []
        for s in r.json().get("data", {}).get("surahs", []):
            for a in s.get("ayahs", []):
                out.append({
                    "surah_no":   s["number"],
                    "surah_name": s["englishName"],
                    "surah_arab": s["name"],
                    "ayah_no":    a["numberInSurah"],
                    "text":       a["text"],
                })
        return out
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner=False)
def load_hadith_book(book_id: str) -> list[dict]:
    try:
        r = requests.get(HADITH_CDN.format(book=book_id), timeout=40)
        r.raise_for_status()
        return r.json().get("hadiths", [])
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DEEP SCORING SEARCH
# Uses word-boundary matching + phrase boosting + root/stem matching
# ══════════════════════════════════════════════════════════════════════════════
def _score_text(text_lower: str, terms: list[str]) -> int:
    score = 0
    for t in terms:
        if " " in t:
            # multi-word phrase — high value
            if t in text_lower:
                score += 4
        else:
            # whole-word boundary match — avoids "pray" matching "prayer" incorrectly
            if re.search(r'\b' + re.escape(t) + r'\b', text_lower):
                score += 2
            elif t in text_lower:
                # partial match — lower value
                score += 1
    return score


def search_quran_deep(terms: list[str]) -> tuple[str, list[dict]]:
    verses = load_full_quran()
    if not verses:
        return "", []

    terms_lower = [t.lower() for t in terms]
    scored = []
    for v in verses:
        s = _score_text(v["text"].lower(), terms_lower)
        if s > 0:
            scored.append((s, v))

    scored.sort(key=lambda x: (-x[0], x[1]["surah_no"], x[1]["ayah_no"]))
    # Return top 15 for display, but only send top 8 to LLM to avoid 413
    top_display = [v for _, v in scored[:15]]
    top_llm     = [v for _, v in scored[:8]]
    return top_display, top_llm


def search_hadith_deep(terms: list[str]) -> tuple[list, list]:
    terms_lower = [t.lower() for t in terms]
    scored: list[tuple[int, str, dict]] = []
    seen:   set[str] = set()

    def _search_book(book_name: str, book_id: str):
        results = []
        for h in load_hadith_book(book_id):
            text = h.get("text", "")
            hid  = f"{book_id}-{h.get('hadithnumber','')}"
            if hid in seen or not text:
                continue
            s = _score_text(text.lower(), terms_lower)
            if s > 0:
                results.append((s, book_name, h, hid))
        return results

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_search_book, bn, bi): bn for bn, bi in HADITH_BOOKS.items()}
        for fut in as_completed(futures):
            for s, bn, h, hid in fut.result():
                if hid not in seen:
                    seen.add(hid)
                    scored.append((s, bn, h))

    scored.sort(key=lambda x: -x[0])
    top_display = [(bn, h) for _, bn, h in scored[:15]]
    top_llm     = [(bn, h) for _, bn, h in scored[:6]]
    return top_display, top_llm


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fmt_quran_display(verses: list[dict]) -> str:
    lines = []
    for v in verses:
        lines.append(
            f"[{v['surah_name']} ({v['surah_arab']}) {v['surah_no']}:{v['ayah_no']}]\n{v['text']}"
        )
    return "\n\n".join(lines)


def fmt_quran_llm(verses: list[dict]) -> str:
    lines = []
    for v in verses:
        lines.append(
            f"Surah {v['surah_name']} ({v['surah_arab']}) — Ch.{v['surah_no']}, V.{v['ayah_no']}:\n\"{v['text']}\""
        )
    return "\n\n".join(lines)


def fmt_hadith_display(items: list[tuple]) -> str:
    lines = []
    for book_name, h in items:
        nar = h.get("by", "")
        nar = f" | Narrated by: {nar}" if nar else ""
        lines.append(f"[{book_name}, #{h.get('hadithnumber','')}]{nar}\n{h.get('text','')}")
    return "\n\n".join(lines)


def fmt_hadith_llm(items: list[tuple]) -> str:
    lines = []
    for book_name, h in items:
        nar  = h.get("by", "")
        nar  = f" — Narrated by: {nar}" if nar else ""
        text = h.get("text", "")
        if len(text) > 500:
            text = text[:500] + "…"
        lines.append(
            f"{book_name}, Hadith #{h.get('hadithnumber','')} {nar}:\n\"{text}\""
        )
    return "\n\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — concise, cited summary answer format
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are a knowledgeable Islamic scholar. Using ONLY the Quran verses and Hadiths provided, give a thorough, well-cited answer.

FORMAT:

## 📖 Quranic Evidence

For each verse, write:
**[Surah Name, Chapter:Verse]** — quote the verse verbatim in italics, then in 2-3 sentences explain exactly what it says about the question.

## 📜 Evidence from Hadith

For each hadith, write:
**[Book, Hadith #N — Narrator (رضي الله عنه)]** — quote the hadith verbatim in italics, then in 2-3 sentences explain its relevance.

## ✅ Summary

In one clear paragraph: state the Islamic ruling/answer with all citations inline like (Surah Al-Baqarah 2:286) and (Sahih Bukhari #6514). Be direct and comprehensive.

RULES:
- Quote every source verbatim (italics) before explaining — no exceptions
- ﷺ after Prophet Muhammad always
- رضي الله عنه/عنها after Companions always  
- Never fabricate anything not in the provided sources
- Cover ALL provided verses and hadiths"""


def ask_groq(question: str, quran_llm: list[dict], hadith_llm: list[tuple]) -> str:
    q_text = fmt_quran_llm(quran_llm)   if quran_llm  else "None found."
    h_text = fmt_hadith_llm(hadith_llm) if hadith_llm else "None found."

    user_msg = (
        f"Question: {question}\n\n"
        f"=== QURAN VERSES (Sahih International) ===\n{q_text}\n\n"
        f"=== HADITH (Bukhari / Muslim / Abu Dawud / Tirmidhi) ===\n{h_text}\n\n"
        f"Answer following the mandatory format. Quote each source verbatim first, then explain. End with a cited summary."
    )
    return _groq(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user",   "content": user_msg}],
        max_tokens=3000,
        temp=0.1,
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

        with st.spinner("🧠 Analysing question…"):
            terms = generate_search_terms(user_input)
        with st.expander(f"🔎 {len(terms)} search dimensions used"):
            st.write(", ".join(terms))

        with st.spinner("📖 Deep searching all 6,236 Quran verses…"):
            q_display, q_llm = search_quran_deep(terms)

        with st.spinner("📜 Deep searching Bukhari, Muslim, Abu Dawud & Tirmidhi…"):
            h_display, h_llm = search_hadith_deep(terms)

        if not q_llm and not h_llm:
            answer = (
                "No relevant verses or Hadiths found. "
                "Please rephrase or ask about a specific topic."
            )
            st.markdown(answer)
        else:
            if q_display:
                with st.expander(f"📖 {len(q_display)} Quran verses found (top 8 used for answer)"):
                    st.text(fmt_quran_display(q_display))
            if h_display:
                with st.expander(f"📜 {len(h_display)} Hadiths found (top 6 used for answer)"):
                    st.text(fmt_hadith_display(h_display))

            with st.spinner("✍️ Composing cited answer…"):
                try:
                    answer = ask_groq(user_input, q_llm, h_llm)
                    if not answer:
                        answer = "⚠️ Groq API key not configured. Add GROQ_API_KEY to Streamlit secrets."
                except Exception as e:
                    answer = f"Error: {e}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<p style="text-align:center;color:#4a7a4a;font-size:0.78rem;margin:0.6rem 0 0.2rem 0;">
  Quran · <a href="https://alquran.cloud">alquran.cloud</a> (Sahih International, all 6,236 verses) &nbsp;·&nbsp;
  Hadith · <a href="https://github.com/fawazahmed0/hadith-api">fawazahmed0/hadith-api</a> (Bukhari · Muslim · Abu Dawud · Tirmidhi) &nbsp;·&nbsp;
  LLM · <a href="https://groq.com">Groq</a> Llama 3.3 70B &nbsp;·&nbsp;
  UI · <a href="https://streamlit.io">Streamlit</a>
</p>
<p style="text-align:center;color:#2a4a2a;font-size:0.72rem;margin:0 0 1rem 0;">All sources free &amp; open. No user data stored.</p>
""", unsafe_allow_html=True)
