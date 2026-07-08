import streamlit as st
import base64
import os
import io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

from utils import extract_text_from_pdf_smart
from llm_functions import (
    summarize_text,
    answer_question,
    compute_full_similarity_matrix_with_original,
    compute_similarity_matrix_api_ninjas_with_original,
)
from pdf_splitter import get_relevant_section

# -------- Load CSS --------
def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")

# -------- Load logo --------
def load_base64_image(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
logo_base64 = load_base64_image("bishops_logo.png")
LOGO_WIDTH = 200

st.markdown(f"""
    <div style='text-align: center; margin-bottom: 1rem;'>
        {"<img src='data:image/png;base64," + logo_base64 + f"' width='{LOGO_WIDTH}'/>" if logo_base64 else ""}
    </div>
    <h1 style='text-align: center; color: #000000;'>🎓 Bishop’s Academic PDF Assistant</h1>
    <h3 style='text-align: center; color: #333;'>Smart Summarization and Q&A</h3>
    <hr style="border:1px solid #582C83;">
""", unsafe_allow_html=True)

# -------- Mode Selector --------
mode = st.radio("Select Mode:", ["Main App (PDF Summarization & Q&A)", "Academic Calendar Q&A"])

# -------- Session state --------
ss = st.session_state
ss.setdefault("summaries", {})
ss.setdefault("answers", {})
ss.setdefault("texts", {})
ss.setdefault("pdf_name", "")
ss.setdefault("best_model", "")
ss.setdefault("best_summary", "")
ss.setdefault("last_cosine_matrix", pd.DataFrame())
ss.setdefault("last_ninjas_matrix", pd.DataFrame())
ss.setdefault("last_sections", [])
ss.setdefault("ninjas_notice", "")

# -------- Models (hidden from users) --------
model_map = {
    "LLaMA 3 - 8B (Meta)": "llama3-8b-8192",
    "LLaMA 3 - 70B (Meta)": "llama3-70b-8192",
    "Hugging Face - LLaMA 2 (13B)": "huggingface-llama2-13b",
}
DEFAULT_MODEL_LABEL = "LLaMA 3 - 8B (Meta)"

# -------- Progress simulation --------
def show_progress():
    progress_bar = st.progress(0)
    for percent in range(0, 101, 10):
        time.sleep(0.08)
        progress_bar.progress(percent)
    progress_bar.empty()

# =========================================================
# ================ MODE 1: MAIN APP =======================
# =========================================================
if mode == "Main App (PDF Summarization & Q&A)":
    st.subheader("📄 Upload a PDF for Summarization & Q&A")
    pdf_file = st.file_uploader("Upload any Bishop’s course-related PDF", type=["pdf"])

    if pdf_file:
        pdf_name = pdf_file.name
        ss.pdf_name = pdf_name

        if pdf_name not in ss.texts:
            with st.spinner("📄 Extracting text..."):
                pdf_bytes = pdf_file.getvalue()
                text, used_ocr, note = extract_text_from_pdf_smart(io.BytesIO(pdf_bytes))
                if used_ocr:
                    st.info("This PDF has no selectable text. I used OCR to read it.")
                if note:
                    st.warning(note)
                if not text.strip():
                    st.error("Couldn’t extract any text from this PDF (even with OCR).")
                ss.texts[pdf_name] = text

        text = ss.texts[pdf_name]

        if st.button("Generate Summary"):
            if not text.strip():
                st.error("No text available to summarize.")
            else:
                st.info("Processing... Please wait.")
                show_progress()
                summaries = {}
                with st.spinner("Summarizing..."):
                    for model_label, model_id in model_map.items():
                        summaries[model_label] = summarize_text(text, model_id)

                cosine_matrix = compute_full_similarity_matrix_with_original(text, summaries)

                ninjas_matrix = None
                ninjas_notice = ""
                try:
                    ninjas_matrix = compute_similarity_matrix_api_ninjas_with_original(text, summaries)
                except Exception as e:
                    ninjas_notice = f"API-Ninjas similarity unavailable: {e}"
                    ninjas_matrix = pd.DataFrame()

                similarity_scores = cosine_matrix.loc["Original PDF"].drop("Original PDF")
                best_model_label = similarity_scores.idxmax()
                best_summary = summaries[best_model_label]

                ss.summaries = summaries
                ss.best_model = best_model_label
                ss.best_summary = best_summary
                ss.last_cosine_matrix = cosine_matrix
                ss.last_ninjas_matrix = ninjas_matrix
                ss.last_sections = []
                ss.ninjas_notice = ninjas_notice

                st.success("Summary generated successfully!")

        if ss.best_summary:
            st.markdown("## 📜 Summary")
            st.write(ss.best_summary.replace("\n", " "))

        st.subheader("💡 Ask a Question")
        question = st.text_input("Enter your question:")
        if question:
            st.info("Processing... Please wait.")
            show_progress()
            chosen_label = ss.best_model if ss.best_model in model_map else DEFAULT_MODEL_LABEL
            model_used = model_map[chosen_label]
            ans_key = f"{pdf_name}__{model_used}__{question}"

            if ans_key not in ss.answers:
                with st.spinner("Generating answer..."):
                    answer = answer_question(ss.texts[pdf_name], question, model_used)
                    ss.answers[ans_key] = answer

            st.markdown("### 💬 Answer:")
            st.write(ss.answers[ans_key].replace("\n", " "))

# =========================================================
# ============== MODE 2: ACADEMIC CALENDAR Q&A ============
# =========================================================
elif mode == "Academic Calendar Q&A":
    st.markdown(
        """
        <h2 style='color:#582C83;'>🎓 Ask Questions About the Bishop's University Academic Calendar (2019–2020)</h2>
        <p style='font-size:16px; color:#444;'>Search across the official Academic Calendar (split into sections). Ask about deadlines, fees, programs, and more.</p>
        """,
        unsafe_allow_html=True,
    )

    calendar_question = st.text_input("Enter your question about the Academic Calendar:")
    if calendar_question:
        st.info("Processing... Please wait.")
        show_progress()
        section_files = get_relevant_section(calendar_question)
        combined_text = ""
        for file in section_files:
            section_path = os.path.join("calendar_sections", file)
            if os.path.exists(section_path):
                with open(section_path, "rb") as f:
                    text, used_ocr, note = extract_text_from_pdf_smart(f)
                    combined_text += text + "\n"

        if combined_text.strip():
            section_summaries = {}
            for model_label, model_id in model_map.items():
                section_summaries[model_label] = summarize_text(combined_text, model_id)

            cosine_matrix = compute_full_similarity_matrix_with_original(combined_text, section_summaries)

            ninjas_matrix = None
            ninjas_notice = ""
            try:
                ninjas_matrix = compute_similarity_matrix_api_ninjas_with_original(combined_text, section_summaries)
            except Exception as e:
                ninjas_notice = f"API-Ninjas similarity unavailable: {e}"
                ninjas_matrix = pd.DataFrame()

            similarity_scores = cosine_matrix.loc["Original PDF"].drop("Original PDF")
            best_model_label = similarity_scores.idxmax()
            best_summary = section_summaries[best_model_label]

            ss.summaries = section_summaries
            ss.best_model = best_model_label
            ss.best_summary = best_summary
            ss.last_cosine_matrix = cosine_matrix
            ss.last_ninjas_matrix = ninjas_matrix
            ss.last_sections = section_files
            ss.ninjas_notice = ninjas_notice

            model_used = model_map[best_model_label]
            ans_key = f"{'_'.join(section_files)}__{model_used}__{calendar_question}"
            if ans_key not in ss.answers:
                with st.spinner("Generating answer..."):
                    answer = answer_question(combined_text, calendar_question, model_used)
                    ss.answers[ans_key] = answer

            st.markdown("### 📜 Summary of Relevant Sections")
            st.write(best_summary.replace("\n", " "))
            st.markdown("### 💬 Answer:")
            st.write(ss.answers[ans_key].replace("\n", " "))
        else:
            st.warning("No relevant section found for this question.")

# ---------------- Expert Sidebar ----------------
with st.sidebar.expander("🔍 Expert Info (Click to Expand)", expanded=False):
    if ss.best_model:
        st.markdown(f"**Chosen Model:** {ss.best_model}")
    if mode == "Academic Calendar Q&A" and ss.last_sections:
        st.markdown(f"**Section(s) Used:** {', '.join(ss.last_sections)}")

    if not ss.last_cosine_matrix.empty:
        st.markdown("**Cosine Similarity Scores:**")
        st.dataframe(ss.last_cosine_matrix, use_container_width=True, height=200)
        st.markdown("**Cosine Similarity Heatmap:**")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.heatmap(ss.last_cosine_matrix, annot=True, cmap="coolwarm", fmt=".4f", vmin=0, vmax=1, ax=ax1)
        st.pyplot(fig1)

    if not ss.last_ninjas_matrix.empty:
        if ss.ninjas_notice:
            st.info(ss.ninjas_notice)
        st.markdown("**API-Ninjas Similarity Scores:**")
        ninj = ss.last_ninjas_matrix.apply(pd.to_numeric, errors="coerce")
        st.dataframe(ninj, use_container_width=True, height=200)
        st.markdown("**API-Ninjas Similarity Heatmap:**")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.heatmap(ninj, annot=True, cmap="coolwarm", fmt=".4f", vmin=0, vmax=1, ax=ax2)
        st.pyplot(fig2)

st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: 12px; color: gray;'>
        Built by Amna, Alpha and Ahmad · CS590 – Bishop’s University · Summer 2025
    </p>
""",
    unsafe_allow_html=True,
)
