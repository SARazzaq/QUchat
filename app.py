import streamlit as st
import requests

# ── Quran.com public API (no key, non-profit, free forever) ───────────────────
QURAN_SEARCH_URL = "https://quran.com/api/v4/search"
QURAN_VERSE_URL  = "https://quran.com/api/v4/verses/by_key/{key}"
# Translation 131 = The Clear Quran by Dr. Mustafa Khattab (authentic, widely used)
TRANSLATION_ID   = 131

# ── Groq API (free tier, no credit card required) ────────────────────────────
GROQ_API_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL       = "llama-3.1-8b-instant"   # fast, free-tier eligible

st.set_page_config(page_title="Quran Chatbot", page_icon="🕌", layout="centered")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        value=st.secrets.get("GROQ_API_KEY", ""),
        help="Free, no credit card needed → https://console.groq.com/keys",
    )
    st.markdown("---")
    st.markdown(
        "**Verse data:** [Quran.com](https://quran.com) public API  \n"
        "**Translation:** The Clear Quran *(Dr. Mustafa Khattab)*  \n"
        "**LLM:** Llama 3.1 via [Groq](https://groq.com) free tier"
    )

# ── Helpers ───────────────────────────────────────────────────────────────────

def search_quran(query: str, num_results: int = 5) -> list[dict]:
    params = {
        "q": query,
        "size": num_results,
        "language": "en",
        "translations": TRANSLATION_ID,
    }
    try:
        r = requests.get(QURAN_SEARCH_URL, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("search", {}).get("results", [])
    except Exception as e:
        st.warning(f"Quran search error: {e}")
        return []


def format_context(results: list[dict]) -> str:
    lines = []
    for r in results:
        key         = r.get("verse_key", "")
        arabic      = r.get("text_uthmani", "")
        translation = ""
        translations = r.get("translations", [])
        if translations:
            translation = translations[0].get("text", "")
        lines.append(f"[{key}]\nArabic: {arabic}\nEnglish: {translation}")
    return "\n\n".join(lines)


SYSTEM_PROMPT = (
    "You are a knowledgeable Islamic assistant specialising in the Quran. "
    "Answer questions strictly based on the Quranic verses provided. "
    "Always cite the verse reference (Surah:Ayah) when quoting or paraphrasing. "
    "If the provided verses do not contain enough information, say so honestly — "
    "never fabricate or invent verses or interpretations. "
    "Use respectful, clear language."
)


def ask_groq(question: str, context: str, api_key: str) -> str:
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
                    f"Relevant Quranic verses:\n\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        "temperature": 0.3,
    }
    r = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🕌 Quran Q&A Chatbot")
st.caption("Ask any question — answers are grounded in authentic Quranic verses.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question about the Quran…"):
    if not groq_key:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching Quran…"):
            results = search_quran(user_input)

        if not results:
            answer = (
                "I could not find relevant verses for your question. "
                "Try rephrasing or ask about a specific topic or verse."
            )
            st.markdown(answer)
        else:
            context = format_context(results)

            with st.expander("📖 Source verses"):
                st.text(context)

            with st.spinner("Generating answer…"):
                try:
                    answer = ask_groq(user_input, context, groq_key)
                except Exception as e:
                    answer = f"Error: {e}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
