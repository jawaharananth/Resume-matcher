# Smart Resume & Job Description Matcher

A tool that compares your resume against a job description and tells you exactly why you match or don't — built because I kept getting rejected for internships and couldn't figure out why.

## What it does

Most resume checkers just give you a score. This one tells you which specific section of your resume is failing, which skills are missing, and which sentences have zero overlap with the job description.

- Section-wise scoring — scores your Experience, Skills, Projects, and Education separately against the JD
- Skill gap analysis — shows exactly which skills are missing and which are matched, grouped by category
- Root cause explainability — tells you why your score is low, not just that it is low
- Weak sentence detector — finds sentences in your resume with no JD overlap and suggests stronger rewrites
- JD sentence breakdown — highlights which requirements you cover and which you miss at sentence level
- Visual report — radar chart and bar chart giving a full picture of your match

## Tech Stack

- Python
- Streamlit for the web interface
- Sentence-BERT (all-MiniLM-L6-v2) for semantic similarity between resume and JD
- SpaCy for skill extraction and NLP processing
- Plotly for interactive visualizations
- Scikit-learn for supporting ML utilities

## How to run it

Clone the repo:
git clone https://github.com/jawaharananth/Resume-matcher.git
cd Resume-matcher

Install dependencies:
pip install streamlit sentence-transformers spacy plotly scikit-learn
python -m spacy download en_core_web_sm

Run the app:
streamlit run app.py

Open http://localhost:8501 in your browser.

## How to use it

1. Paste your resume text on the left
2. Paste the job description on the right
3. Click Analyze Match
4. Go through each tab to see your section scores, skill gaps, weak sentences, and JD breakdown

## Screenshots

Add screenshots here after running the app

## What I learned

Building this made me understand how semantic similarity actually works differently from keyword matching. Sentence-BERT captures meaning while TF-IDF just counts words — the difference shows up clearly when your resume talks about the same thing using different terminology than the JD.

## Future improvements

- Add transformer-based sentence rewriting instead of rule-based
- Support PDF resume upload directly
- Add ATS compatibility checker