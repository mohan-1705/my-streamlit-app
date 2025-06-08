import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.lower()

# Streamlit UI
st.title("🧠 AI Resume Screening Tool")
st.write("Upload a job description and multiple resumes (PDFs) to find the best match.")

# Upload job description
job_desc_file = st.file_uploader("Upload Job Description (TXT)", type=["txt"])
resumes = st.file_uploader("Upload Resumes (PDFs)", type=["pdf"], accept_multiple_files=True)

if job_desc_file and resumes:
    job_description = job_desc_file.read().decode("utf-8").lower()

    results = []
    for resume in resumes:
        # Read PDF resume
        resume_text = extract_text_from_pdf(resume)

        # Vectorization
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([job_description, resume_text])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        match_score = round(similarity * 100, 2)

        results.append((resume.name, match_score))

    # Sort results by match
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    st.subheader("📊 Resume Match Scores")
    for name, score in sorted_results:
        st.write(f"**{name}**: {score}% match")
