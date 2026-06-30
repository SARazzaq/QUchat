import streamlit as st
import requests

# ── Constants ─────────────────────────────────────────────────────────────────
QURAN_SEARCH  = "https://api.alquran.cloud/v1/search/{query}/all/en.sahih"
GROQ_URL      = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama-3.3-70b-versatile"   # best free model on Groq

HADITH_BOOKS  = {
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

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #f0ece4;
}
.stApp {
    background: #0f1e0f;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Chat input ── */
.stChatInput textarea {
    background: #1a2e1a !important;
    color: #f5f0e8 !important;
    border: 1.5px solid #b8960c !important;
    border-radius: 14px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
}
.stChatInput textarea::placeholder { color: #7a9a7a !important; }

/* ── Chat messages ── */
.stChatMessage {
    background: transparent !important;
}
[data-testid="stChatMessageContent"] {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    line-height: 1.75;
    color: #f0ece4;
}

/* ── Expander ── */
details {
    background: #152015 !important;
    border: 1px solid #2a4a2a !important;
    border-radius: 10px !important;
}
summary {
    color: #d4a017 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 14px !important;
}
.streamlit-expanderContent {
    color: #c8dcc0 !important;
    font-size: 0.85rem !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.6 !important;
}

/* ── Info box (disclaimer) ── */
[data-testid="stAlert"] {
    background: #1a2a1a !important;
    border-left: 4px solid #d4a017 !important;
    border-radius: 10px !important;
    color: #e8dfc8 !important;
    font-size: 0.92rem !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #d4a017 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0f1e0f; }
::-webkit-scrollbar-thumb { background: #b8960c; border-radius: 3px; }

/* ── Arabic text ── */
.arabic {
    font-family: 'Amiri', serif;
    font-size: 1.4rem;
    color: #f5e6a0;
    direction: rtl;
    line-height: 2.2;
}

/* ── Divider ── */
hr { border-color: #1e3a1e !important; }

/* ── Links ── */
a { color: #7ec88a !important; }
a:hover { color: #d4a017 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2.5rem 0 1.2rem 0;">
  <div style="font-size:3.2rem; margin-bottom:0.3rem;">🕌</div>
  <h1 style="font-family:'Amiri',serif; color:#d4a017; font-size:2.8rem; margin:0; letter-spacing:1px;">
    QUchat
  </h1>
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

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.info(
    "⚠️ **Disclaimer:** Answers are AI-generated and grounded in authentic Islamic sources. "
    "AI can make mistakes — always verify important matters with a qualified Islamic scholar. "
    "Quran: Sahih International · Hadith: Sahih Bukhari & Sahih Muslim."
)

# ── API helpers ───────────────────────────────────────────────────────────────

def search_quran(query: str) -> tuple[str, bool]:
    """Search Quran via alquran.cloud. Returns (formatted_context, found)."""
    try:
        url = QURAN_SEARCH.format(query=requests.utils.quote(query))
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        matches = r.json().get("data", {}).get("matches", [])[:6]
        if not matches:
            return "", False
        lines = []
        for m in matches:
            surah_name = m.get("surah", {}).get("englishName", "")
            surah_no   = m.get("surah", {}).get("number", "")
            ayah_no    = m.get("numberInSurah", "")
            text       = m.get("text", "")
            lines.append(f"[Surah {surah_name} {surah_no}:{ayah_no}]\n{text}")
        return "\n\n".join(lines), True
    except Exception:
        return "", False


def search_hadith(query: str) -> tuple[str, bool]:
    """Keyword search in Bukhari and Muslim. Returns (formatted_context, found)."""
    keywords = [w.lower() for w in query.split() if len(w) > 3]
    results  = []
    for book_name, book_id in HADITH_BOOKS.items():
        try:
            r = requests.get(HADITH_CDN.format(book=book_id), timeout=15)
            r.raise_for_status()
            hadiths = r.json().get("hadiths", [])
            count = 0
            for h in hadiths:
                text = h.get("text", "")
                if any(kw in text.lower() for kw in keywords):
                    results.append(
                        f"[{book_name}, Hadith #{h.get('hadithnumber', '')}]\n{text}"
                    )
                    count += 1
                    if count >= 2:
                        break
        except Exception:
            continue
    return "\n\n".join(results[:4]), bool(results)


SYSTEM_PROMPT = """You are a deeply knowledgeable Islamic scholar assistant. Your role is to give thorough, well-referenced answers based strictly on the Quran and authentic Hadith provided.

INSTRUCTIONS:
1. Give elaborate, complete answers — do not be brief.
2. For every Quranic point, cite the full reference: Surah name and number, Ayah number. Example: (Surah Al-Baqarah 2:255).
3. For every Hadith point, cite: book name and hadith number. Example: (Sahih Bukhari, Hadith #8).
4. Quote the actual verse or hadith text before explaining it.
5. Structure your answer with clear sections when covering multiple points.
6. If the context doesn't fully answer the question, say so clearly — never invent or fabricate.
7. Write with respect and care. Add ﷺ after Prophet Muhammad's name every time.
8. End with a brief summary of the key Islamic ruling or guidance."""


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
                    f"Use the following authentic Islamic sources to answer the question.\n\n"
                    f"--- SOURCES ---\n{context}\n--- END SOURCES ---\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=45)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ── Chat state ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask about the Quran or Islam…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Step 1: Quran search
        with st.spinner("🔍 Searching Quran…"):
            quran_ctx, quran_found = search_quran(user_input)

        # Step 2: Hadith search (always run — supplements Quran results)
        with st.spinner("📜 Searching Hadith…"):
            hadith_ctx, hadith_found = search_hadith(user_input)

        # Build combined context
        sections = []
        if quran_found:
            sections.append(f"=== QURAN (Sahih International) ===\n{quran_ctx}")
        if hadith_found:
            sections.append(f"=== HADITH (Sahih Bukhari / Sahih Muslim) ===\n{hadith_ctx}")
        combined = "\n\n".join(sections)

        if not combined:
            answer = (
                "I could not find relevant Quranic verses or Hadiths for your question. "
                "Please try rephrasing, or ask about a specific topic or verse."
            )
            st.markdown(answer)
        else:
            # Show sources
            if quran_found:
                with st.expander("📖 Quran verses retrieved"):
                    st.text(quran_ctx)
            if hadith_found:
                with st.expander("📜 Hadith retrieved"):
                    st.text(hadith_ctx)

            with st.spinner("✍️ Composing answer…"):
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
  All sources are free &amp; open. No user data is stored.
</p>
""", unsafe_allow_html=True)
