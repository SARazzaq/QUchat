import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# Primary   : alquran.cloud  (Islamic Network — open, no key)
# Fallback  : islamic.app    (Cloudflare edge, 600 req/min, no key)
# Hadith P  : fawazahmed0/hadith-api via jsDelivr CDN (no key, no rate limit)
# Hadith FB : islamic.app /v1/hadith/search  (50k hadiths, full-text, no key)
# ══════════════════════════════════════════════════════════════════════════════

ALQURAN_SEARCH    = "https://api.alquran.cloud/v1/search/{q}/all/en.sahih"
ALQURAN_VERSE     = "https://api.alquran.cloud/v1/ayah/{key}/en.sahih"
ISLAMICAPP_SEARCH = "https://api.islamic.app/v1/search"
ISLAMICAPP_HADITH = "https://api.islamic.app/v1/hadith/search"

HADITH_CDN   = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/{book}.json"
HADITH_BOOKS = {"Sahih Bukhari": "eng-bukhari", "Sahih Muslim": "eng-muslim"}

GROQ_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT CONFIG
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
    font-family:'Inter',sans-serif; font-size:1rem; line-height:1.85; color:#f0ece4;
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
    "Quran: Sahih International · Hadith: Sahih Bukhari & Sahih Muslim."
)

# ══════════════════════════════════════════════════════════════════════════════
# QUERY EXPANSION
# ══════════════════════════════════════════════════════════════════════════════

SYNONYMS = {
    "namaz":["salah","prayer","salat","worship","prostrate","bow","establish"],
    "salah":["namaz","prayer","salat","worship","prostrate"],
    "prayer":["salah","namaz","salat","worship","dua","supplication"],
    "salat":["salah","namaz","prayer"],
    "roza":["fasting","sawm","fast","ramadan"],
    "fasting":["roza","sawm","fast","ramadan","hunger"],
    "sawm":["fasting","roza","fast"],
    "zakat":["charity","alms","zakah","give","poor","needy"],
    "hajj":["pilgrimage","makkah","kabah","kaaba"],
    "jihad":["struggle","striving","fight","effort"],
    "mandatory":["obligatory","fard","compulsory","wajib","required","duty"],
    "obligatory":["mandatory","fard","compulsory","wajib"],
    "forbidden":["haram","prohibited","unlawful","disallowed"],
    "haram":["forbidden","prohibited","unlawful"],
    "halal":["permissible","allowed","lawful"],
    "heaven":["paradise","jannah","garden"],
    "paradise":["heaven","jannah","garden","afterlife"],
    "jannah":["paradise","heaven","garden"],
    "hell":["jahannam","fire","punishment","torment"],
    "quran":["revelation","scripture","book","verses"],
    "prophet":["muhammad","messenger","rasool","nabi"],
    "muhammad":["prophet","messenger","rasool","nabi"],
    "allah":["god","lord","creator","rabb"],
    "sin":["transgression","evil","wrong","disobey"],
    "repentance":["tawbah","forgiveness","tawba","mercy"],
    "tawbah":["repentance","forgiveness"],
    "death":["dying","afterlife","akhirah","soul"],
    "akhirah":["afterlife","hereafter","death","resurrection"],
    "alcohol":["wine","khamr","intoxicant","drinking","liquor"],
    "interest":["riba","usury","bank"],
    "riba":["interest","usury"],
    "marriage":["nikah","wife","husband","spouse","wed"],
    "nikah":["marriage","wed","spouse"],
    "divorce":["talaq","separation","wife"],
    "talaq":["divorce","separation"],
    "women":["female","mother","wife","sister","hijab"],
    "hijab":["veil","cover","women","modesty","purdah"],
    "charity":["sadaqah","zakat","give","poor","needy","help"],
    "sadaqah":["charity","give","poor"],
    "parents":["mother","father","family","obedience","honor"],
    "justice":["adl","fairness","equity","rights"],
    "patience":["sabr","endure","trial","hardship"],
    "sabr":["patience","endure"],
    "knowledge":["ilm","learn","education","wisdom"],
    "ilm":["knowledge","learn","education"],
    "food":["eat","halal","haram","meat","pork","pig"],
    "pork":["pig","swine","forbidden","haram","food"],
}


def expand_queries(question: str) -> list[str]:
    q      = question.lower().strip()
    tokens = [w.strip("?.,!;:'\"") for w in q.split() if len(w.strip("?.,!;:'\"")) > 2]
    queries: set[str] = set()

    # raw tokens
    for t in tokens:
        queries.add(t)
        for syn in SYNONYMS.get(t, []):
            queries.add(syn)

    # bigrams
    for i in range(len(tokens) - 1):
        queries.add(f"{tokens[i]} {tokens[i+1]}")

    # full question
    queries.add(question)

    # topic-level short queries (3-word max)
    meaningful = [t for t in tokens if len(t) > 3]
    for t in meaningful[:5]:
        queries.add(t)

    return list(queries)[:15]


# ══════════════════════════════════════════════════════════════════════════════
# QURAN SEARCH  (primary: alquran.cloud  |  fallback: islamic.app)
# ══════════════════════════════════════════════════════════════════════════════

def _alquran_search_one(q: str) -> list[dict]:
    try:
        r = requests.get(
            ALQURAN_SEARCH.format(q=requests.utils.quote(q)),
            timeout=8
        )
        if r.status_code == 200:
            return r.json().get("data", {}).get("matches", [])
    except Exception:
        pass
    return []


def _islamicapp_search(q: str) -> list[dict]:
    """Fallback: islamic.app /v1/search — returns quran.com-compatible verses."""
    try:
        r = requests.get(
            ISLAMICAPP_SEARCH,
            params={"q": q, "size": 10, "translations": "20"},  # 20 = Sahih International
            timeout=8,
        )
        if r.status_code == 200:
            data = r.json()
            # islamic.app mirrors quran.com shape
            results = (
                data.get("search", {}).get("results", [])
                or data.get("results", [])
                or []
            )
            hits = []
            for res in results:
                verse_key = res.get("verse_key", "")
                if not verse_key:
                    continue
                parts = verse_key.split(":")
                if len(parts) != 2:
                    continue
                surah_no, ayah_no = parts
                hits.append({
                    "surah": {
                        "number": int(surah_no),
                        "englishName": res.get("surah_name_english", f"Surah {surah_no}"),
                        "name": res.get("surah_name", ""),
                    },
                    "numberInSurah": int(ayah_no),
                    "text": (res.get("translations") or [{}])[0].get("text", "")
                            or res.get("text", ""),
                    "_source": "islamic.app",
                })
            return hits
    except Exception:
        pass
    return []


def search_quran(question: str) -> tuple[str, bool]:
    queries = expand_queries(question)
    seen: set[str] = set()
    hits: list[dict] = []

    def fetch(q):
        return _alquran_search_one(q)

    # Run all queries in parallel
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch, q): q for q in queries}
        for fut in as_completed(futures):
            for m in fut.result():
                sn = m.get("surah", {}).get("number", 0)
                an = m.get("numberInSurah", 0)
                key = f"{sn}:{an}"
                if key not in seen:
                    seen.add(key)
                    hits.append(m)

    # If primary returned nothing, try islamic.app fallback
    if not hits:
        for q in queries[:5]:
            for m in _islamicapp_search(q):
                sn = m.get("surah", {}).get("number", 0)
                an = m.get("numberInSurah", 0)
                key = f"{sn}:{an}"
                if key not in seen:
                    seen.add(key)
                    hits.append(m)

    if not hits:
        return "", False

    hits.sort(key=lambda m: (m.get("surah", {}).get("number", 0), m.get("numberInSurah", 0)))

    lines = []
    for m in hits[:15]:
        sname  = m.get("surah", {}).get("englishName", "")
        sarab  = m.get("surah", {}).get("name", "")
        sno    = m.get("surah", {}).get("number", "")
        ano    = m.get("numberInSurah", "")
        text   = m.get("text", "")
        src    = m.get("_source", "alquran.cloud")
        lines.append(
            f"[Surah {sname} ({sarab}) — Chapter {sno}, Verse {ano}] [{src}]\n{text}"
        )

    return "\n\n".join(lines), True


# ══════════════════════════════════════════════════════════════════════════════
# HADITH SEARCH  (primary: fawazahmed0 CDN  |  fallback: islamic.app)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _load_hadith_book(book_id: str) -> list[dict]:
    """Cache the full hadith book in memory for fast keyword search."""
    try:
        r = requests.get(HADITH_CDN.format(book=book_id), timeout=25)
        r.raise_for_status()
        return r.json().get("hadiths", [])
    except Exception:
        return []


def _islamicapp_hadith_search(keyword: str) -> list[dict]:
    """Fallback: islamic.app full-text hadith search."""
    try:
        r = requests.get(
            ISLAMICAPP_HADITH,
            params={"q": keyword, "collections": "bukhari,muslim", "limit": 4},
            timeout=8,
        )
        if r.status_code == 200:
            return r.json().get("results", [])
    except Exception:
        pass
    return []


def search_hadith(question: str) -> tuple[str, bool]:
    queries  = expand_queries(question)
    keywords = [q.lower() for q in queries if len(q.split()) == 1 and len(q) > 3]
    seen:    set[str] = set()
    results: list[str] = []

    # Primary: search cached CDN books
    for book_name, book_id in HADITH_BOOKS.items():
        hadiths = _load_hadith_book(book_id)
        count   = 0
        for h in hadiths:
            text = h.get("text", "")
            hid  = f"{book_id}-{h.get('hadithnumber','')}"
            if hid in seen:
                continue
            if any(kw in text.lower() for kw in keywords):
                seen.add(hid)
                narrator = h.get("by", "")
                nar_str  = f" — Narrated by: {narrator}" if narrator else ""
                results.append(
                    f"[{book_name}, Hadith #{h.get('hadithnumber','')}{nar_str}]\n{text}"
                )
                count += 1
                if count >= 4:
                    break

    # Fallback: islamic.app hadith search if primary found nothing
    if not results:
        for kw in keywords[:3]:
            for h in _islamicapp_hadith_search(kw):
                coll = h.get("collection", "")
                num  = h.get("hadithNumber", "")
                text = h.get("text", "") or h.get("body", "")
                hid  = f"islamic.app-{coll}-{num}"
                if hid not in seen and text:
                    seen.add(hid)
                    results.append(f"[{coll.title()}, Hadith #{num} — islamic.app]\n{text}")

    return "\n\n".join(results[:8]), bool(results)


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a deeply knowledgeable Islamic scholar. Your task is to give thorough, properly referenced answers using ONLY the Quran and Hadith sources provided.

═══════════ MANDATORY ANSWER STRUCTURE ═══════════

## 📖 From the Quran

For EVERY verse in the context:

**Surah [Name] ([Arabic Name]) — Chapter [X], Verse [Y]**

> [Paste the COMPLETE verbatim text of the verse exactly as given — do not shorten or paraphrase]

**Explanation:** Write a detailed explanation of this verse in the context of the question. Cover:
- What Allah ﷻ is commanding, forbidding, or informing
- The significance and depth of this specific verse
- How it directly answers the question
- Any important nuance in the wording

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📜 From the Hadith

For EVERY Hadith in the context:

**[Book Name], Hadith #[Number] — Narrated by: [Narrator] (رضي الله عنه/عنها)**

> [Paste the COMPLETE verbatim text of the Hadith exactly as given — do not shorten or paraphrase]

**Explanation:** Write a detailed explanation of this Hadith in the context of the question. Cover:
- What the Prophet ﷺ said, did, or approved
- The ruling or guidance it establishes
- How it complements or elaborates on the Quranic verses above
- Its importance for the question asked

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## ✅ Islamic Ruling & Summary

Write a complete, clear Islamic ruling that:
1. States whether the matter is Fard (فرض obligatory), Sunnah (مستحب recommended), Haram (حرام forbidden), or Mubah (مباح permissible)
2. Summarises the combined evidence from both Quran and Hadith above
3. Explains the wisdom behind this ruling in Islam

═══════════════════════════════════════════

ABSOLUTE RULES — never break these:
1. ALWAYS paste the full verbatim text of every verse and hadith in blockquote (>) before explaining
2. Cover EVERY verse and EVERY hadith in the context — skip none
3. Write ﷺ after every mention of Prophet Muhammad
4. Write رضي الله عنه / رضي الله عنها after every Companion's name
5. Never fabricate, invent, or add anything not present in the provided sources
6. Be thorough — long detailed answers are required
7. Read the question from every possible angle before answering"""


def ask_groq(question: str, context: str) -> str:
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return "⚠️ Groq API key not configured. Add GROQ_API_KEY to your Streamlit secrets."
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"QUESTION: {question}\n\n"
                    f"Analyse this question from every angle — linguistic, theological, and practical.\n\n"
                    f"{'═'*55}\n"
                    f"AUTHENTIC ISLAMIC SOURCES\n"
                    f"{'═'*55}\n\n"
                    f"{context}\n\n"
                    f"{'═'*55}\n\n"
                    f"Now write a full, detailed, properly referenced answer "
                    f"following the mandatory structure. Quote every source verbatim first, then explain."
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


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
            verse_count  = quran_ctx.count("[Surah ") if quran_found else 0
            hadith_count = hadith_ctx.count(", Hadith #") if hadith_found else 0

            if quran_found:
                with st.expander(f"📖 {verse_count} Quran verse(s) retrieved"):
                    st.text(quran_ctx)
            if hadith_found:
                with st.expander(f"📜 {hadith_count} Hadith retrieved"):
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
  Quran · <a href="https://alquran.cloud">alquran.cloud</a> + <a href="https://islamic.app">islamic.app</a> (Sahih International) &nbsp;·&nbsp;
  Hadith · <a href="https://github.com/fawazahmed0/hadith-api">fawazahmed0</a> + <a href="https://islamic.app">islamic.app</a> (Bukhari & Muslim) &nbsp;·&nbsp;
  LLM · <a href="https://groq.com">Groq</a> Llama 3.3 70B &nbsp;·&nbsp;
  UI · <a href="https://streamlit.io">Streamlit</a>
</p>
<p style="text-align:center; color:#2a4a2a; font-size:0.72rem; margin:0 0 1rem 0;">
  All sources free &amp; open. No user data stored.
</p>
""", unsafe_allow_html=True)
