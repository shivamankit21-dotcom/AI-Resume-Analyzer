import os 
import streamlit as st
from pypdf import PdfReader
 
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("🧠 AI Resume Analyzer & Job Recommender")
st.write("Upload your resume and get AI-powered job suggestions")

# Resume text extraction
def extract_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text.lower()


# Clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# Load job roles
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "sample_jobs.txt")

with open(file_path, "r", encoding="utf-8") as f:
    jobs = f.readlines()


job_titles = []
job_descriptions = []

for job in jobs:
    if ":" in job:
        title, desc = job.split(":", 1)
        job_titles.append(title.strip())
        job_descriptions.append(desc.strip())


uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    resume_text = extract_text(uploaded_file)
    resume_text = clean_text(resume_text)

    documents = job_descriptions + [resume_text]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    scores = similarity[0]

    st.subheader("✅ Recommended Job Roles")

    results = sorted(zip(job_titles, scores), key=lambda x: x[1], reverse=True)

    for role, score in results[:3]:
        st.write(f"**{role}**")
        st.progress(float(score))

    st.subheader("📊 Resume Strength Score")
    resume_score = int(max(scores) * 100)
    st.metric("Score", f"{resume_score}/100")

    if resume_score > 70:
        st.success("Strong Resume 💪")
    elif resume_score > 40:
        st.warning("Average Resume ⚠️")
    else:
        st.error("Needs Improvement ❌")



# RUN THIS LINE IN TERMINAL TO EXCUTE THE PROGRAM :- C:/Users/shiva/AppData/Local/Python/pythoncore-3.14-64/python.exe -m streamlit run C:/Users/shiva/OneDrive/Desktop/hackthon.py

