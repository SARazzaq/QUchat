import streamlit as st
import requests

# ── API endpoints ─────────────────────────────────────────────────────────────
# Quran: alquran.cloud — free, no auth, Sahih International (authentic)
QURAN_SEARCH  = "https://api.alquran.cloud/v1/search/{query}/all/en.sahih"
QURAN_ARABIC  = "https://api.alquran.cloud/v1/search/{query}/all/quran-uthmani"

# Hadith: fawazahmed0/hadith-api via jsDelivr CDN — free, no key, no rate limit
# Books: Sahih Bukhari and Sahih Muslim (most authentic — Kutub al-Sittah)
HADITH_SEARCH_URL = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/{book}.json"
HADITH_BOOKS = ["eng-bukhari", "eng-muslim"]

# Groq — free tier, no credit card
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="QUchat — Quran Q&A", page_icon="🕌", layout="centered")

# ── Islamic UI styling ────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&family=Lato:wght@300;400;700&display=swap');

  /* Background */
  .stApp {
    background: linear-gradient(160deg, #0d1b2a 0%, #1a2f1a 60%, #0d1b2a 100%);
    color: #e8dcc8;
    font-family: 'Lato', sans-serif;
  }

  /* Hide default streamlit header/footer */
  #MainMenu, footer, header { visibility: hidden; }

  /* Chat input */
  .stChatInput textarea {
    background: #1e3a2f !important;
    color: #e8dcc8 !important;
    border: 1px solid #c9a84c !important;
    border-radius: 12px !important;
  }

  /* User message bubble */
  [data-testid="stChatMessageContent"]:has(+ div [data-testid="stChatMessageAvatarUser"]),
  .stChatMessage[data-testid*="user"] [data-testid="stChatMessageContent"] {
    background: #1e3a2f;
    border-left: 3px solid #c9a84c;
    border-radius: 12px;
    padding: 12px 16px;
  }

  /* Assistant message bubble */
  .stChatMessage[data-testid*="assistant"] [data-testid="stChatMessageContent"] {
    background: #162512;
    border-left: 3px solid #4caf7d;
    border-radius: 12px;
    padding: 12px 16px;
  }

  /* Expander */
  .streamlit-expanderHeader {
    background: #1a2f1a !important;
    color: #c9a84c !important;
    border-radius: 8px !important;
  }
  .streamlit-expanderContent {
    background: #111f11 !important;
    border-radius: 0 0 8px 8px !important;
    font-size: 0.85rem;
    color: #b5c8a8;
  }

  /* Spinner */
  .stSpinner > div { border-top-color: #c9a84c !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0d1b2a; }
  ::-webkit-scrollbar-thumb { background: #c9a84c; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem 0;">
  <div style="font-size:3rem;">🕌</div>
  <h1 style="font-family:'Amiri',serif; color:#c9a84c; font-size:2.4rem; margin:0;">
    QUchat
  </h1>
  <p style="color:#8fad88; font-size:1rem; margin-top:0.3rem; letter-spacing:1px;">
    Quran & Hadith Q&A — Grounded in Authentic Islamic Sources
  </p>
  <p style="font-family:'Amiri',serif; color:#c9a84c; font-size:1.3rem; margin-top:0.5rem;">
    بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
  </p>
</div>
<hr style="border-color:#2a4a2a; margin-bottom:1rem;">
""", unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.info(
    "⚠️ **Disclaimer:** This chatbot uses AI to interpret Quranic verses and Hadiths. "
    "AI can make mistakes — always verify with a qualified Islamic scholar. "
    "Quran text is Sahih International; Hadiths are from Sahih Bukhari & Sahih Muslim."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def search_quran(query: str) -> tuple[str, list[dict]]:
    """Returns (formatted_context, raw_results). Uses Sahih International."""
    try:
        r = requests.get(QURAN_SEARCH.format(query=requests.utils.quote(query)), timeout=10)
        r.raise_for_status()
        data = r.json()
        matches = data.get("data", {}).get("matches", [])[:5]
        if not matches:
            return "", []
        lines = []
        for m in matches:
            ref  = m.get("surah", {}).get("englishName", "") + " " + str(m.get("numberInSurah", ""))
            key  = f"{m.get('surah', {}).get('number', '')}:{m.get('numberInSurah', '')}"
            text = m.get("text", "")
            lines.append(f"[{key}] {ref}\n{text}")
        return "\n\n".join(lines), matches
    except Exception:
        return "", []


def search_hadith(query: str) -> str:
    """Keyword search across Bukhari and Muslim. Returns formatted context."""
    keywords = [w.lower() for w in query.split() if len(w) > 3]
    results = []
    for book in HADITH_BOOKS:
        try:
            url = HADITH_SEARCH_URL.format(book=book)
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            hadiths = r.json().get("hadiths", [])
            book_label = "Sahih Bukhari" if "bukhari" in book else "Sahih Muslim"
            count = 0
            for h in hadiths:
                text = h.get("text", "").lower()
                if any(kw in text for kw in keywords):
                    results.append(
                        f"[{book_label} #{h.get('hadithnumber', '')}]\n{h.get('text', '')}"
                    )
                    count += 1
                    if count >= 2:
                        break
        except Exception:
            continue
    return "\n\n".join(results[:4])


SYSTEM_PROMPT = """You are a knowledgeable Islamic assistant. Answer questions using ONLY the Quranic verses and Hadiths provided as context.

Rules:
- Always cite the reference (Surah:Ayah or Hadith book/number).
- If context is insufficient, say so — never fabricate verses or Hadiths.
- Prefer Quran over Hadith when both are available.
- Use respectful, clear language. Add "ﷺ" after Prophet Muhammad's name."""


def ask_groq(question: str, context: str) -> str:
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return "⚠️ Groq API key not configured. Please add GROQ_API_KEY to your Streamlit secrets."
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}"},
        ],
        "temperature": 0.2,
    }
    r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ── Chat UI ───────────────────────────────────────────────────────────────────
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
        # 1. Search Quran
        with st.spinner("Searching Quran…"):
            quran_context, quran_matches = search_quran(user_input)

        # 2. If Quran didn't return enough, also search Hadith
        hadith_context = ""
        if not quran_matches:
            with st.spinner("Searching Hadith…"):
                hadith_context = search_hadith(user_input)

        combined_context = ""
        if quran_context:
            combined_context += f"=== Quran (Sahih International) ===\n{quran_context}"
        if hadith_context:
            combined_context += f"\n\n=== Hadith (Sahih Bukhari / Sahih Muslim) ===\n{hadith_context}"

        if not combined_context:
            answer = (
                "I could not find relevant verses or Hadiths for your question. "
                "Please try rephrasing, or ask about a specific topic or verse reference."
            )
            st.markdown(answer)
        else:
            if quran_context:
                with st.expander("📖 Quran verses used"):
                    st.text(quran_context)
            if hadith_context:
                with st.expander("📜 Hadith used"):
                    st.text(hadith_context)

            with st.spinner("Generating answer…"):
                try:
                    answer = ask_groq(user_input, combined_context)
                except Exception as e:
                    answer = f"Error contacting AI: {e}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#2a4a2a; margin-top:2rem;">
<p style="text-align:center; color:#4a6a4a; font-size:0.78rem; margin:0.5rem 0;">
  Quran text · <a href="https://alquran.cloud" style="color:#6a9a6a;">alquran.cloud</a> (Sahih International) &nbsp;|&nbsp;
  Hadith · <a href="https://github.com/fawazahmed0/hadith-api" style="color:#6a9a6a;">fawazahmed0/hadith-api</a> (Bukhari & Muslim) &nbsp;|&nbsp;
  AI · <a href="https://groq.com" style="color:#6a9a6a;">Groq</a> Llama 3.1 &nbsp;|&nbsp;
  Built with <a href="https://streamlit.io" style="color:#6a9a6a;">Streamlit</a>
</p>
<p style="text-align:center; color:#3a5a3a; font-size:0.72rem; margin:0.2rem 0 1rem 0;">
  All sources are free & open. No user data is stored.
</p>
""", unsafe_allow_html=True)
