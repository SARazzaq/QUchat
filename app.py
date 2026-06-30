import streamlit as st
import requests

# ── Constants ─────────────────────────────────────────────────────────────────
QURAN_SEARCH  = "https://api.alquran.cloud/v1/search/{query}/all/en.sahih"
GROQ_URL      = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama-3.3-70b-versatile"

HADITH_BOOKS = {
    "Sahih Bukhari": "eng-bukhari",
    "Sahih Muslim":  "eng-muslim",
}
HADITH_CDN = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/{book}.json"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="QUchat", page_icon="🕌", layout="centered")

# ── CSS ───────────────────────────────────────────────────────────────────────
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
    font-family:'Inter',sans-serif; font-size:1rem;
    line-height:1.8; color:#f0ece4;
}
details {
    background:#152015 !important; border:1px solid #2a4a2a !important;
    border-radius:10px !important;
}
summary {
    color:#d4a017 !important; font-weight:600 !important;
    font-size:0.9rem !important; padding:10px 14px !important;
}
.streamlit-expanderContent {
    color:#c8dcc0 !important; font-size:0.85rem !important;
    font-family:'Inter',sans-serif !important; line-height:1.6 !important;
}
[data-testid="stAlert"] {
    background:#1a2a1a !important; border-left:4px solid #d4a017 !important;
    border-radius:10px !important; color:#e8dfc8 !important; font-size:0.92rem !important;
}
.stSpinner > div { border-top-color:#d4a017 !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#0f1e0f; }
::-webkit-scrollbar-thumb { background:#b8960c; border-radius:3px; }
.arabic {
    font-family:'Amiri',serif; font-size:1.4rem;
    color:#f5e6a0; direction:rtl; line-height:2.2;
}
hr { border-color:#1e3a1e !important; }
a { color:#7ec88a !important; }
a:hover { color:#d4a017 !important; }
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
    "AI can make mistakes — always verify important matters with a qualified Islamic scholar. "
    "Quran: Sahih International · Hadith: Sahih Bukhari & Sahih Muslim."
)

# ── Search helpers ────────────────────────────────────────────────────────────

def expand_queries(question: str) -> list[str]:
    """
    Break the question into multiple search angles so we cast a wide net.
    E.g. 'Is namaz mandatory?' → ['namaz', 'prayer', 'salah', 'obligatory', 'fard']
    """
    q = question.lower().strip()
    tokens = [w.strip("?.,!") for w in q.split() if len(w.strip("?.,!")) > 3]

    # Common Islamic synonym map
    synonyms = {
        "namaz": ["salah", "prayer", "salat"],
        "salah": ["namaz", "prayer", "salat"],
        "prayer": ["salah", "namaz", "salat"],
        "salat": ["salah", "namaz", "prayer"],
        "roza": ["fasting", "sawm", "fast"],
        "fasting": ["roza", "sawm", "fast"],
        "sawm": ["fasting", "roza", "fast"],
        "zakat": ["charity", "alms", "zakah"],
        "hajj": ["pilgrimage", "makkah"],
        "jihad": ["struggle", "striving"],
        "mandatory": ["obligatory", "fard", "compulsory", "wajib"],
        "obligatory": ["mandatory", "fard", "compulsory", "wajib"],
        "forbidden": ["haram", "prohibited", "unlawful"],
        "haram": ["forbidden", "prohibited", "unlawful"],
        "halal": ["permissible", "allowed", "lawful"],
        "heaven": ["paradise", "jannah"],
        "paradise": ["heaven", "jannah"],
        "jannah": ["paradise", "heaven"],
        "hell": ["jahannam", "fire", "punishment"],
        "quran": ["quran", "revelation", "scripture"],
        "prophet": ["muhammad", "messenger", "rasool"],
        "muhammad": ["prophet", "messenger", "rasool"],
        "allah": ["god", "lord", "creator"],
        "sin": ["gunah", "transgression", "evil"],
        "repentance": ["tawbah", "forgiveness", "tawba"],
        "tawbah": ["repentance", "forgiveness"],
        "death": ["dying", "afterlife", "akhirah"],
        "akhirah": ["afterlife", "hereafter", "death"],
    }

    queries = set()
    # Original tokens as individual queries
    for t in tokens:
        queries.add(t)
        if t in synonyms:
            queries.update(synonyms[t])

    # Full question as one query
    queries.add(question)

    # Pairs of adjacent meaningful tokens
    for i in range(len(tokens) - 1):
        queries.add(f"{tokens[i]} {tokens[i+1]}")

    return list(queries)[:10]   # cap at 10 search calls


def search_quran(question: str) -> tuple[str, bool]:
    """Multi-angle Quran search. Deduplicates by verse key."""
    queries  = expand_queries(question)
    seen     = set()
    all_hits = []

    for q in queries:
        try:
            url = QURAN_SEARCH.format(query=requests.utils.quote(q))
            r   = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue
            for m in r.json().get("data", {}).get("matches", []):
                surah_no = m.get("surah", {}).get("number", "")
                ayah_no  = m.get("numberInSurah", "")
                key      = f"{surah_no}:{ayah_no}"
                if key not in seen:
                    seen.add(key)
                    all_hits.append(m)
        except Exception:
            continue

    if not all_hits:
        return "", False

    # Sort by surah then ayah for clean presentation
    all_hits.sort(key=lambda m: (
        m.get("surah", {}).get("number", 0),
        m.get("numberInSurah", 0)
    ))

    lines = []
    for m in all_hits[:12]:   # up to 12 unique verses
        surah_name   = m.get("surah", {}).get("englishName", "")
        surah_arabic = m.get("surah", {}).get("name", "")
        surah_no     = m.get("surah", {}).get("number", "")
        ayah_no      = m.get("numberInSurah", "")
        text         = m.get("text", "")
        lines.append(
            f"[Surah {surah_name} ({surah_arabic}) — Chapter {surah_no}, Verse {ayah_no}]\n"
            f"{text}"
        )

    return "\n\n".join(lines), True


def search_hadith(question: str) -> tuple[str, bool]:
    """Multi-keyword Hadith search across Bukhari and Muslim."""
    queries  = expand_queries(question)
    keywords = [q.lower() for q in queries if len(q.split()) == 1]  # single-word only for matching
    seen     = set()
    results  = []

    for book_name, book_id in HADITH_BOOKS.items():
        try:
            r = requests.get(HADITH_CDN.format(book=book_id), timeout=20)
            r.raise_for_status()
            hadiths = r.json().get("hadiths", [])
            count   = 0
            for h in hadiths:
                text    = h.get("text", "")
                h_id    = f"{book_id}-{h.get('hadithnumber', '')}"
                if h_id in seen:
                    continue
                if any(kw in text.lower() for kw in keywords):
                    seen.add(h_id)
                    narrator = h.get("by", "") or ""
                    narrator_str = f" — Narrated by {narrator}" if narrator else ""
                    results.append(
                        f"[{book_name}, Hadith #{h.get('hadithnumber', '')}{narrator_str}]\n{text}"
                    )
                    count += 1
                    if count >= 3:
                        break
        except Exception:
            continue

    return "\n\n".join(results[:6]), bool(results)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a deeply knowledgeable Islamic scholar. Your job is to give thorough, properly referenced answers using ONLY the Quran verses and Hadith provided in the context.

═══════════════════════════════════════
MANDATORY ANSWER STRUCTURE — every response must follow this exactly:
═══════════════════════════════════════

## 📖 Answer from the Quran

For EVERY relevant verse provided:

**Surah [Name] ([Arabic Name]) — Chapter [X], Verse [Y]**

> "[Paste the EXACT full text of the verse here — do not paraphrase]"

**Explanation:** Elaborate on what Allah is saying in this verse in the context of the question. Explain the meaning deeply. What does it command, forbid, or inform? How does it directly answer the question? Give full detail.

[Repeat for every relevant verse]

---

## 📜 Answer from the Hadith

For EVERY relevant Hadith provided:

**[Book Name], Hadith #[Number] — Narrated by [Narrator name] (رضي الله عنه)**

> "[Paste the EXACT full text of the Hadith here — do not paraphrase]"

**Explanation:** Elaborate on what the Prophet ﷺ is saying or doing. How does this Hadith answer the question? What ruling or guidance does it give? Give full detail.

[Repeat for every relevant Hadith]

---

## ✅ Islamic Ruling & Summary

Combine all the above into a clear, complete Islamic ruling. State what is obligatory (فرض), recommended (مستحب), forbidden (حرام), or permissible (حلال) with reasons drawn from the sources above.

═══════════════════════════════════════
STRICT RULES:
═══════════════════════════════════════
1. ALWAYS paste the exact verbatim text of every verse and hadith in blockquote (>) before explaining it.
2. Never skip a verse or hadith from the context — cover ALL of them.
3. Always write ﷺ after Prophet Muhammad's name.
4. Always write رضي الله عنه / رضي الله عنها after a Companion's name.
5. Never fabricate, paraphrase, or add anything not in the provided sources.
6. Be thorough — long detailed answers are required, not brief ones.
7. Read the question from multiple angles before answering — address all dimensions of it."""


def ask_groq(question: str, context: str) -> str:
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return "⚠️ Groq API key not configured. Add GROQ_API_KEY to your Streamlit secrets."
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"QUESTION: {question}\n\n"
                    f"Read this question carefully from every angle. "
                    f"Now use ONLY the sources below to give a complete answer.\n\n"
                    f"{'='*60}\n"
                    f"AUTHENTIC SOURCES\n"
                    f"{'='*60}\n\n"
                    f"{context}\n\n"
                    f"{'='*60}\n\n"
                    f"Now write a full, detailed, properly referenced answer following the mandatory structure."
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 4000,
    }
    r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ── Chat ──────────────────────────────────────────────────────────────────────
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
        with st.spinner("🔍 Searching Quran across multiple angles…"):
            quran_ctx, quran_found = search_quran(user_input)

        with st.spinner("📜 Searching Hadith…"):
            hadith_ctx, hadith_found = search_hadith(user_input)

        sections = []
        if quran_found:
            sections.append(f"=== QURAN (Sahih International Translation) ===\n\n{quran_ctx}")
        if hadith_found:
            sections.append(f"=== HADITH (Sahih Bukhari / Sahih Muslim) ===\n\n{hadith_ctx}")
        combined = "\n\n".join(sections)

        if not combined:
            answer = (
                "I could not find relevant Quranic verses or Hadiths for your question. "
                "Please try rephrasing, or ask about a specific topic or verse."
            )
            st.markdown(answer)
        else:
            if quran_found:
                with st.expander(f"📖 Quran verses retrieved ({quran_ctx.count('[Surah')} verses)"):
                    st.text(quran_ctx)
            if hadith_found:
                with st.expander(f"📜 Hadith retrieved ({hadith_ctx.count('[Sahih')} hadiths)"):
                    st.text(hadith_ctx)

            with st.spinner("✍️ Composing detailed answer…"):
                try:
                    answer = ask_groq(user_input, combined)
                except Exception as e:
                    answer = f"Error contacting AI: {e}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<p style="text-align:center; color:#4a7a4a; font-size:0.78rem; margin:0.6rem 0 0.2rem 0;">
  Quran · <a href="https://alquran.cloud">alquran.cloud</a> (Sahih International) &nbsp;·&nbsp;
  Hadith · <a href="https://github.com/fawazahmed0/hadith-api">fawazahmed0/hadith-api</a> (Bukhari & Muslim) &nbsp;·&nbsp;
  LLM · <a href="https://groq.com">Groq</a> Llama 3.3 70B &nbsp;·&nbsp;
  UI · <a href="https://streamlit.io">Streamlit</a>
</p>
<p style="text-align:center; color:#2a4a2a; font-size:0.72rem; margin:0 0 1rem 0;">
  All sources free &amp; open. No user data stored.
</p>
""", unsafe_allow_html=True)
