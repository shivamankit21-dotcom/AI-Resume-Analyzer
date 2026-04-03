import os
import re
import streamlit as st
from pypdf import PdfReader
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("🧠 AI Resume Analyzer & Job Recommender")
st.write("Upload your resume, select your skills, and check your best matching job roles.")

# ---------------------------
# FUNCTION TO EXTRACT TEXT FROM PDF
# ---------------------------
def extract_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text.lower()

# ---------------------------
# CLEAN TEXT
# ---------------------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z+ ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# ---------------------------
# LOAD JOB DATA FROM sample_jobs.txt
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "sample_jobs.txt")

job_titles = []
job_descriptions = []

with open(file_path, "r", encoding="utf-8") as f:
    jobs = f.readlines()

for job in jobs:
    if ":" in job:
        title, desc = job.split(":", 1)
        job_titles.append(title.strip())
        job_descriptions.append(desc.strip().lower())

# ---------------------------
# SKILL OPTIONS FOR USER INPUT
# ---------------------------
skill_options = [
    "C", "C++", "Python", "Java", "JavaScript", "HTML", "CSS", "SQL", "R",
    "Pandas", "NumPy", "Machine Learning", "Deep Learning", "TensorFlow",
    "PyTorch", "Django", "Flask", "React", "Node.js", "Kotlin", "Android",
    "Firebase", "Data Analysis", "Statistics", "API", "Database",
    "Communication", "Problem Solving", "Git", "GitHub", "OOP", "DSA",
    "Neural Networks", "Frontend", "Backend", "SEO", "Blogging",
    "Copywriting", "Research", "Recruitment", "HR Policies", "Management",
    "Planning", "Reporting", "CRM", "Sales", "Branding", "Documentation"
]

# ---------------------------
# USER INPUTS
# ---------------------------
selected_skills = st.multiselect(
    "💡 Select the languages / skills you know",
    skill_options
)

preferred_job = st.selectbox(
    "🎯 Select your preferred job profession",
    job_titles
)

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

# ---------------------------
# MAIN LOGIC
# ---------------------------
if uploaded_file:
    resume_text = extract_text(uploaded_file)
    resume_text = clean_text(resume_text)

    # Selected skills text
    selected_skills_text = " ".join(selected_skills).lower()

    # Final combined text
    final_resume_text = resume_text + " " + selected_skills_text

    # TF-IDF similarity for job recommendation
    documents = job_descriptions + [final_resume_text]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    scores = similarity[0]

    # ---------------------------
    # JOB RECOMMENDATIONS
    # ---------------------------
    st.subheader("✅ Recommended Job Roles")

    results = sorted(zip(job_titles, scores), key=lambda x: x[1], reverse=True)

    for role, score in results[:3]:
        st.write(f"**{role}**")
        st.progress(float(score))
        st.write(f"Match Score: {round(score * 100, 2)}%")

    # ---------------------------
    # RESUME STRENGTH SCORE
    # ---------------------------
    st.subheader("📊 Resume Strength Score")
    resume_score = int(max(scores) * 100)
    st.metric("Score", f"{resume_score}/100")

    if resume_score > 70:
        st.success("Strong Resume 💪")
    elif resume_score > 40:
        st.warning("Average Resume ⚠️")
    else:
        st.error("Needs Improvement ❌")

    # ---------------------------
    # SHOW SELECTED SKILLS
    # ---------------------------
    if selected_skills:
        st.subheader("🛠 Selected Skills")
        st.write(", ".join(selected_skills))

    # ---------------------------
    # MISSING SKILLS FOR PREFERRED JOB
    # ---------------------------
    st.subheader("🎯 Preferred Job Analysis")
    st.write(f"**Preferred Profession:** {preferred_job}")

    preferred_job_index = job_titles.index(preferred_job)
    preferred_job_desc = job_descriptions[preferred_job_index]

    required_skills = [skill.strip().lower() for skill in preferred_job_desc.split(",")]

    # Normalize selected skills for comparison
    normalized_selected_skills = set(skill.lower() for skill in selected_skills)

    matched_skills = []
    missing_skills = []

    for skill in required_skills:
        if skill in final_resume_text or skill in normalized_selected_skills:
            matched_skills.append(skill.title())
        else:
            missing_skills.append(skill.title())

    if matched_skills:
        st.write("✅ Skills you already have:")
        st.write(", ".join(matched_skills))

    if missing_skills:
        st.write("❌ Missing skills you should learn:")
        st.write(", ".join(missing_skills))
        st.warning(
            f"To become a strong {preferred_job}, you should learn: {', '.join(missing_skills)}"
        )
    else:
        st.success(f"Great! You already have most of the important skills for {preferred_job}.")




