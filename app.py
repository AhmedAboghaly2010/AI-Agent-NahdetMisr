# ============================================================
#  🤖 RAG System — نظام الذكاء الاصطناعي للأسئلة والأجوبة
#  
#  RAG = Retrieval Augmented Generation
#  الفكرة البسيطة:
#    1. ارفع ملفاتك (PDF / Excel / CSV)
#    2. النظام يقرأها ويقسمها لـ "قطع صغيرة" (Chunks)
#    3. لما تسأل سؤال، النظام يلاقي القطع الأقرب لسؤالك
#    4. يبعتها لـ Gemini مع سؤالك → يجيب إجابة ذكية
# ============================================================

import streamlit as st          # إطار عمل الواجهة
import google.generativeai as genai  # مكتبة Gemini
import pandas as pd              # قراءة Excel و CSV
import pdfplumber                # قراءة PDF
import numpy as np               # عمليات رياضية على الـ Vectors
import io, os, json, time

# ══════════════════════════════════════════════════════════
#  إعداد الصفحة
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Agent — نظام الأسئلة الذكي",
    page_icon="🤖",
    layout="wide",
)

# ── CSS (ثيم داكن + عربي) ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');

:root {
  --bg: #0a0a1a; --surface: #12122a; --card: #1a1a35;
  --border: #2a2a55; --accent: #6366f1; --accent2: #22d3ee;
  --text: #e2e8f0; --muted: #64748b; --green: #34d399; --red: #f87171;
}
html, body, [class*="css"] { font-family: 'Cairo', sans-serif !important; }
.stApp { background: var(--bg); color: var(--text); }
section[data-testid="stSidebar"] { background: var(--surface); border-left: 1px solid var(--border); }

/* Chat bubbles */
.msg-user {
  background: linear-gradient(135deg, #4f46e5, #6366f1);
  color: white; border-radius: 18px 18px 4px 18px;
  padding: 12px 18px; margin: 8px 0 8px 60px;
  text-align: right; direction: rtl; line-height: 1.7;
}
.msg-ai {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 18px 18px 18px 4px;
  padding: 14px 18px; margin: 8px 60px 8px 0;
  text-align: right; direction: rtl; line-height: 1.9;
}
.msg-ai-label { color: var(--accent2); font-size: 0.75rem; font-weight: 700; margin-bottom: 6px; }
.msg-user-label { color: rgba(255,255,255,0.7); font-size: 0.75rem; margin-bottom: 6px; text-align: left; }

/* Source cards */
.source-card {
  background: var(--surface); border: 1px solid var(--border);
  border-right: 3px solid var(--accent2);
  border-radius: 8px; padding: 10px 14px;
  margin: 4px 0; font-size: 0.8rem; color: var(--muted);
  direction: rtl; text-align: right;
}

/* KPI */
.kpi { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  padding: 16px; text-align: center; }
.kpi-val { font-size: 1.8rem; font-weight: 900; color: var(--accent2); }
.kpi-lbl { font-size: 0.8rem; color: var(--muted); margin-top: 4px; }

/* Status */
.status-ok  { color: var(--green); font-weight: 700; }
.status-err { color: var(--red);   font-weight: 700; }

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), #4f46e5) !important;
  color: white !important; border: none !important; border-radius: 10px !important;
  font-family: 'Cairo', sans-serif !important; font-weight: 700 !important;
  width: 100%;
}
div[data-testid="stTextInput"] input {
  background: var(--card) !important; color: var(--text) !important;
  border-color: var(--border) !important; font-family: 'Cairo', sans-serif !important;
  direction: rtl; text-align: right;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  📚 الخطوة 1: قراءة الملفات واستخراج النصوص
#  كل نوع ملف له طريقة قراءة مختلفة
# ══════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes):
    """
    قراءة نص من ملف PDF
    pdfplumber بتفتح كل صفحة وتستخرج النص منها
    """
    text_pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():  # تجاهل الصفحات الفاضية
                text_pages.append({
                    "content": text.strip(),
                    "source": f"صفحة {i+1}"
                })
    return text_pages


def extract_text_from_table(file_bytes, filename):
    """
    قراءة بيانات من Excel أو CSV
    بنحول الجدول لنص وصفي عشان الـ AI يفهمه
    """
    if filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))

    chunks = []

    # ── معلومات عامة عن الجدول ──
    summary = f"""
    ملخص الجدول: {filename}
    عدد الصفوف: {len(df):,}
    عدد الأعمدة: {len(df.columns)}
    الأعمدة: {', '.join(df.columns.tolist())}
    
    إحصائيات سريعة:
    {df.describe(include='all').to_string()}
    """
    chunks.append({"content": summary, "source": f"{filename} — ملخص"})

    # ── تقسيم الجدول لـ chunks صغيرة (كل 50 صف) ──
    # السبب: الـ AI مش بيقدر يقرأ آلاف الصفوف مرة واحدة
    chunk_size = 50
    for start in range(0, len(df), chunk_size):
        chunk_df = df.iloc[start : start + chunk_size]
        chunks.append({
            "content": chunk_df.to_string(index=False),
            "source": f"{filename} — صفوف {start+1} إلى {start+len(chunk_df)}"
        })

    return chunks


# ══════════════════════════════════════════════════════════
#  ✂️ الخطوة 2: تقسيم النصوص لـ Chunks
#  السبب: الـ AI عنده حد أقصى للنص اللي يقدر يقرأه
#  فبنقسم النص لقطع صغيرة متداخلة شوية (Overlap)
#  الـ Overlap مهم عشان المعلومات اللي على حدود القطعتين متضيعش
# ══════════════════════════════════════════════════════════

def split_into_chunks(text, source, chunk_size=800, overlap=100):
    """
    chunk_size = حجم كل قطعة بالحروف
    overlap    = عدد الحروف المشتركة بين قطعتين متجاورتين
    """
    chunks = []
    words = text.split()           # قسّم النص لكلمات
    current_chunk = []
    current_size = 0

    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1  # +1 للمسافة

        if current_size >= chunk_size:
            chunk_text = " ".join(current_chunk)
            chunks.append({"content": chunk_text, "source": source})

            # ← ابدأ القطعة الجديدة من آخر N كلمة (الـ Overlap)
            overlap_words = current_chunk[-overlap//5:]
            current_chunk = overlap_words
            current_size = sum(len(w) + 1 for w in overlap_words)

    # أضف باقي النص لو فيه
    if current_chunk:
        chunks.append({"content": " ".join(current_chunk), "source": source})

    return chunks


# ══════════════════════════════════════════════════════════
#  🔢 الخطوة 3: تحويل النص لـ Embedding (Vector)
#  
#  الـ Embedding هو تحويل النص لأرقام (Vector)
#  مثلاً: "الربح" → [0.2, -0.5, 0.8, 0.1, ...]
#
#  الهدف: نص متشابه في المعنى → أرقام متشابهة
#  فنقدر نلاقي أقرب chunk لسؤال المستخدم رياضياً
# ══════════════════════════════════════════════════════════

def get_embedding(text, model):
    """
    بنبعت النص لـ Gemini وبيرجع لنا Vector (قائمة أرقام)
    """
    result = genai.embed_content(
        model=model,
        content=text,
        task_type="retrieval_document"  # نوع المهمة: فهرسة نصوص
    )
    return np.array(result["embedding"])


def cosine_similarity(vec1, vec2):
    """
    قياس التشابه بين Vectorين
    النتيجة بين 0 (مختلفان تماماً) و 1 (متطابقان)
    
    الفكرة: زي قياس الزاوية بين خطين في الفضاء
    كلما الزاوية صغيرة → كلما النصين أقرب في المعنى
    """
    dot_product = np.dot(vec1, vec2)          # حاصل الضرب النقطي
    norm1 = np.linalg.norm(vec1)              # طول الـ Vector الأول
    norm2 = np.linalg.norm(vec2)              # طول الـ Vector الثاني
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)      # التشابه


# ══════════════════════════════════════════════════════════
#  🔍 الخطوة 4: البحث عن أقرب Chunks للسؤال
#  
#  الخطوات:
#  1. حوّل السؤال لـ Vector
#  2. قيس تشابهه مع كل Chunk في قاعدة البيانات
#  3. خذ أعلى N نتائج
# ══════════════════════════════════════════════════════════

def retrieve_relevant_chunks(question, vector_db, embed_model, top_k=4):
    """
    top_k = عدد الـ Chunks اللي هنبعتهم لـ Gemini
    """
    # حوّل السؤال لـ Vector
    question_vec = genai.embed_content(
        model=embed_model,
        content=question,
        task_type="retrieval_query"   # نوع المهمة: بحث (مختلف عن الفهرسة)
    )["embedding"]
    question_vec = np.array(question_vec)

    # احسب التشابه مع كل Chunk
    scored = []
    for chunk in vector_db:
        score = cosine_similarity(question_vec, np.array(chunk["embedding"]))
        scored.append({**chunk, "score": score})

    # رتّب من الأعلى للأقل تشابهاً
    scored.sort(key=lambda x: x["score"], reverse=True)

    return scored[:top_k]


# ══════════════════════════════════════════════════════════
#  💬 الخطوة 5: توليد الإجابة بـ Gemini
#  
#  بنبني "Prompt" يحتوي على:
#  1. تعليمات للـ AI (System Prompt)
#  2. النصوص المسترجعة كـ "سياق"
#  3. سؤال المستخدم
# ══════════════════════════════════════════════════════════

def generate_answer(question, relevant_chunks, chat_history, gemini_model):
    """
    بنبني الـ Prompt ونبعته لـ Gemini
    """
    # ── بناء السياق من الـ Chunks المسترجعة ──
    context_parts = []
    for i, chunk in enumerate(relevant_chunks, 1):
        context_parts.append(
            f"[مصدر {i}: {chunk['source']} | تشابه: {chunk['score']:.0%}]\n{chunk['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # ── System Prompt: تعليمات للـ AI ──
    system_prompt = """أنت مساعد ذكي متخصص في تحليل البيانات وتقارير الأعمال.
مهمتك: الإجابة على أسئلة المستخدم بناءً على البيانات المقدمة فقط.

قواعد مهمة:
- استخدم فقط المعلومات الموجودة في السياق المقدم
- لو السؤال مش موجود في البيانات، قول ذلك بوضوح
- قدّم الأرقام بشكل منظم مع جداول لو مناسب
- الإجابة بالعربية دائماً ما لم يسأل المستخدم بلغة أخرى
- كن دقيقاً ومختصراً"""

    # ── بناء تاريخ المحادثة ──
    history_text = ""
    for msg in chat_history[-4:]:   # آخر 4 رسائل فقط (توفير tokens)
        role = "المستخدم" if msg["role"] == "user" else "المساعد"
        history_text += f"{role}: {msg['content']}\n"

    # ── الـ Prompt النهائي ──
    full_prompt = f"""{system_prompt}

══ السياق المسترجع من ملفاتك ══
{context}

══ تاريخ المحادثة ══
{history_text}

══ السؤال الحالي ══
{question}

الإجابة:"""

    # ── إرسال لـ Gemini والحصول على الإجابة ──
    response = gemini_model.generate_content(full_prompt)
    return response.text


# ══════════════════════════════════════════════════════════
#  🗄️ إدارة الـ Session State
#  
#  Streamlit بيعيد تشغيل الكود كل مرة المستخدم يعمل حاجة
#  Session State بيخلينا نحتفظ بالبيانات بين كل run
# ══════════════════════════════════════════════════════════

if "vector_db"     not in st.session_state: st.session_state.vector_db     = []
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "files_loaded"  not in st.session_state: st.session_state.files_loaded  = []
if "gemini_ready"  not in st.session_state: st.session_state.gemini_ready  = False


# ══════════════════════════════════════════════════════════
#  🎨 الواجهة — Sidebar
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:20px 0; border-bottom:1px solid #2a2a55; margin-bottom:20px;">
      <div style="font-size:2.5rem;">🤖</div>
      <div style="font-size:1.1rem; font-weight:700; color:#6366f1;">AI Agent</div>
      <div style="font-size:0.75rem; color:#64748b;">نظام الأسئلة الذكي</div>
    </div>
    """, unsafe_allow_html=True)

    # ── إعداد API Key ──
    st.markdown("### 🔑 Gemini API Key")
    api_key = st.text_input("أدخل الـ API Key", type="password",
                             placeholder="AIzaSy...")

    if api_key:
        try:
            genai.configure(api_key=api_key)
            # اختبار الاتصال
            _ = genai.GenerativeModel("gemini-1.5-flash")
            st.markdown('<p class="status-ok">✅ الاتصال ناجح</p>', unsafe_allow_html=True)
            st.session_state.gemini_ready = True
            st.session_state.api_key = api_key
        except Exception as e:
            st.markdown(f'<p class="status-err">❌ خطأ: {str(e)[:60]}</p>', unsafe_allow_html=True)
            st.session_state.gemini_ready = False

    st.markdown("---")

    # ── رفع الملفات ──
    st.markdown("### 📁 ارفع ملفاتك")
    uploaded_files = st.file_uploader(
        "PDF / Excel / CSV",
        type=["pdf", "xlsx", "xls", "csv"],
        accept_multiple_files=True,
        help="ارفع ملف واحد أو أكثر"
    )

    if uploaded_files and st.session_state.gemini_ready:
        if st.button("🚀 معالجة الملفات"):
            # ── نموذج الـ Embedding ──
            EMBED_MODEL = "models/gemini-embedding-001"
            progress = st.progress(0, text="جاري المعالجة...")
            all_chunks = []

            for file_idx, file in enumerate(uploaded_files):
                file_bytes = file.read()
                file_name  = file.name

                progress.progress(
                    (file_idx) / len(uploaded_files),
                    text=f"📄 جاري قراءة: {file_name}"
                )

                # ── استخراج النص حسب نوع الملف ──
                raw_chunks = []
                if file_name.endswith(".pdf"):
                    pages = extract_text_from_pdf(file_bytes)
                    for page in pages:
                        # قسّم كل صفحة لـ chunks أصغر
                        sub_chunks = split_into_chunks(
                            page["content"], page["source"]
                        )
                        raw_chunks.extend(sub_chunks)

                else:  # Excel أو CSV
                    raw_chunks = extract_text_from_table(file_bytes, file_name)

                # ── توليد الـ Embeddings ──
                # هنا البطء ← كل chunk بيتبعت لـ Gemini منفرداً
                for i, chunk in enumerate(raw_chunks):
                    try:
                        embedding = get_embedding(chunk["content"], EMBED_MODEL)
                        chunk["embedding"] = embedding.tolist()
                        all_chunks.append(chunk)
                        time.sleep(0.1)  # تأخير بسيط لتجنب Rate Limit
                    except Exception as e:
                        st.warning(f"⚠️ تخطي chunk: {str(e)[:40]}")

                st.session_state.files_loaded.append(file_name)

            # حفظ قاعدة البيانات في الـ Session
            st.session_state.vector_db = all_chunks
            progress.progress(1.0, text="✅ اكتملت المعالجة!")
            st.success(f"✅ تم تحميل {len(all_chunks)} قطعة نصية")
            time.sleep(1)
            st.rerun()

    st.markdown("---")

    # ── إحصائيات ──
    if st.session_state.vector_db:
        st.markdown("### 📊 الإحصائيات")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="kpi">
            <div class="kpi-val">{len(st.session_state.vector_db)}</div>
            <div class="kpi-lbl">Chunks</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="kpi">
            <div class="kpi-val">{len(st.session_state.files_loaded)}</div>
            <div class="kpi-lbl">ملفات</div></div>""", unsafe_allow_html=True)

        st.markdown("**الملفات المحملة:**")
        for f in st.session_state.files_loaded:
            st.markdown(f"- 📄 {f}")

    # ── مسح المحادثة ──
    if st.button("🗑️ مسح كل شيء"):
        for key in ["vector_db", "chat_history", "files_loaded"]:
            st.session_state[key] = [] if isinstance(st.session_state[key], list) else {}
        st.rerun()


# ══════════════════════════════════════════════════════════
#  🎨 الواجهة الرئيسية — Chat
# ══════════════════════════════════════════════════════════

st.markdown("""
<div style="background:linear-gradient(135deg,#12122a,#1a1a35); border:1px solid #2a2a55;
     border-radius:16px; padding:24px 32px; margin-bottom:24px; text-align:right;">
  <div style="font-size:0.85rem; color:#64748b; margin-bottom:4px;">
    RAG — Retrieval Augmented Generation
  </div>
  <div style="font-size:1.8rem; font-weight:900;
    background:linear-gradient(90deg,#6366f1,#22d3ee);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    🤖 المساعد الذكي لبياناتك
  </div>
  <div style="color:#64748b; font-size:0.9rem; margin-top:6px;">
    ارفع ملفاتك ← اسأل أي سؤال ← احصل على إجابة مباشرة من بياناتك
  </div>
</div>
""", unsafe_allow_html=True)

# ── عرض الـ Chat History ──
chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        # رسالة ترحيب
        st.markdown("""
        <div class="msg-ai">
          <div class="msg-ai-label">🤖 المساعد الذكي</div>
          مرحباً! أنا مساعدك الذكي 👋<br><br>
          <b>كيف أعمل؟</b><br>
          ① ارفع ملفاتك (PDF أو Excel أو CSV) من القائمة الجانبية<br>
          ② اضغط "معالجة الملفات" وانتظر<br>
          ③ اسألني أي سؤال عن بياناتك!<br><br>
          <b>أمثلة على أسئلة:</b><br>
          • "ما إجمالي المبيعات في 2024؟"<br>
          • "هل الشركة حققت ربح أم خسارة؟"<br>
          • "ما أكثر المنتجات مبيعاً؟"<br>
          • "لخّص لي التقرير المالي"
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">
              <div class="msg-user-label">أنت</div>
              {msg["content"]}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-ai">
              <div class="msg-ai-label">🤖 المساعد الذكي</div>
              {msg["content"]}
            </div>""", unsafe_allow_html=True)

            # عرض المصادر المستخدمة
            if "sources" in msg:
                with st.expander("📎 المصادر المستخدمة في الإجابة"):
                    for src in msg["sources"]:
                        st.markdown(f"""
                        <div class="source-card">
                          📄 <b>{src['source']}</b> &nbsp;|&nbsp;
                          تشابه: <b>{src['score']:.0%}</b><br>
                          <span style="color:#94a3b8;">{src['content'][:150]}...</span>
                        </div>""", unsafe_allow_html=True)

# ── مربع السؤال ──
st.markdown("<br>", unsafe_allow_html=True)
col_input, col_btn = st.columns([5, 1])

with col_input:
    question = st.text_input(
        "سؤالك",
        placeholder="اسأل أي سؤال عن بياناتك...",
        label_visibility="collapsed",
        key="question_input"
    )

with col_btn:
    send = st.button("إرسال ✈️")

# ── أسئلة سريعة ──
st.markdown("<div style='text-align:right; color:#64748b; font-size:0.8rem; margin:8px 0 4px;'>أسئلة سريعة:</div>",
            unsafe_allow_html=True)
q_cols = st.columns(4)
quick_questions = [
    "📊 لخّص البيانات",
    "💰 ما إجمالي الإيرادات؟",
    "📈 ما أفضل أداء؟",
    "⚠️ أي مشاكل في البيانات؟"
]
quick_q = None
for i, (col, q) in enumerate(zip(q_cols, quick_questions)):
    with col:
        if st.button(q, key=f"quick_{i}"):
            quick_q = q.split(" ", 1)[1]  # حذف الإيموجي


# ══════════════════════════════════════════════════════════
#  🧠 المنطق الرئيسي: معالجة السؤال
# ══════════════════════════════════════════════════════════

final_question = question if send and question else quick_q

if final_question:
    # ── التحقق من الشروط ──
    if not st.session_state.gemini_ready:
        st.error("❌ أدخل الـ API Key أولاً")
        st.stop()

    if not st.session_state.vector_db:
        st.error("❌ ارفع ملفات أولاً وافضل معالجتها")
        st.stop()

    # ── إعداد نماذج Gemini ──
    genai.configure(api_key=st.session_state.api_key)
    EMBED_MODEL = "models/gemini-embedding-001"
    GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")

    with st.spinner("🔍 جاري البحث في ملفاتك..."):

        # الخطوة 1: ابحث عن أقرب Chunks
        relevant = retrieve_relevant_chunks(
            final_question,
            st.session_state.vector_db,
            EMBED_MODEL,
            top_k=4
        )

        # الخطوة 2: ولّد الإجابة
        answer = generate_answer(
            final_question,
            relevant,
            st.session_state.chat_history,
            GEMINI_MODEL
        )

    # ── حفظ في الـ History ──
    st.session_state.chat_history.append({
        "role": "user",
        "content": final_question
    })
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": relevant
    })

    st.rerun()


# ── Footer ──
st.markdown("""
<div style="text-align:center; padding:24px; color:#334155; font-size:0.78rem;
     border-top:1px solid #1e1e3f; margin-top:32px;">
  🤖 AI Agent — RAG System &nbsp;•&nbsp; Powered by Google Gemini &nbsp;•&nbsp;
  Built with Streamlit
</div>
""", unsafe_allow_html=True)
