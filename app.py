import streamlit as st
from sentence_transformers import SentenceTransformer, util
import spacy
import re
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    nlp = spacy.load('en_core_web_sm')
    return model, nlp

model, nlp = load_models()

SKILL_DATABASE = {
    "Programming Languages": [
        "python", "java", "javascript", "c++", "c#", "ruby", "go", "rust",
        "typescript", "scala", "kotlin", "swift", "r", "matlab", "php", "c"
    ],
    "Machine Learning & AI": [
        "machine learning", "deep learning", "neural networks", "nlp",
        "natural language processing", "computer vision", "reinforcement learning",
        "transformer", "bert", "gpt", "llm", "generative ai", "pytorch",
        "tensorflow", "keras", "scikit-learn", "xgboost", "random forest",
        "gradient boosting", "feature engineering", "model deployment"
    ],
    "Data & Analytics": [
        "data analysis", "data science", "pandas", "numpy", "matplotlib",
        "seaborn", "tableau", "power bi", "sql", "nosql", "mongodb",
        "postgresql", "mysql", "data visualization", "statistics", "excel"
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "jenkins",
        "git", "github", "linux", "bash", "terraform", "ansible", "mlflow"
    ],
    "Web Development": [
        "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
        "html", "css", "rest api", "graphql", "microservices", "spring boot"
    ],
    "Soft Skills": [
        "communication", "leadership", "teamwork", "problem solving",
        "critical thinking", "project management", "agile", "scrum", "research"
    ]
}

def extract_resume_sections(resume_text):
    sections = {
        "Experience": "",
        "Education": "",
        "Skills": "",
        "Projects": "",
        "Achievements": ""
    }
    lines = resume_text.split('\n')
    current_section = "General"
    section_content = defaultdict(list)
    
    section_keywords = {
        "Experience": ["experience", "work history", "employment", "internship", "intern"],
        "Education": ["education", "academic", "qualification", "degree", "university", "college"],
        "Skills": ["skills", "technical skills", "technologies", "tools", "competencies"],
        "Projects": ["projects", "portfolio", "works", "implementations"],
        "Achievements": ["achievements", "awards", "certifications", "accomplishments", "honors"]
    }
    
    for line in lines:
        line_lower = line.lower().strip()
        matched_section = None
        for section, keywords in section_keywords.items():
            if any(kw in line_lower for kw in keywords):
                matched_section = section
                break
        if matched_section:
            current_section = matched_section
        else:
            section_content[current_section].append(line)
    
    for section in sections:
        sections[section] = ' '.join(section_content.get(section, [])).strip()
    
    return sections

def extract_skills(text):
    text_lower = text.lower()
    found_skills = defaultdict(list)
    for category, skills in SKILL_DATABASE.items():
        for skill in skills:
            if skill in text_lower:
                found_skills[category].append(skill)
    return found_skills

def compute_semantic_score(text1, text2):
    if not text1.strip() or not text2.strip():
        return 0.0
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2)
    return float(score[0][0]) * 100

def compute_section_scores(sections, jd_text):
    scores = {}
    for section_name, section_content in sections.items():
        if section_content.strip():
            score = compute_semantic_score(section_content, jd_text)
            scores[section_name] = round(score, 1)
        else:
            scores[section_name] = 0.0
    return scores

def get_skill_gap_analysis(resume_skills, jd_skills):
    missing = defaultdict(list)
    matched = defaultdict(list)
    all_resume_skills = [s for skills in resume_skills.values() for s in skills]
    
    for category, skills in jd_skills.items():
        for skill in skills:
            if skill in all_resume_skills:
                matched[category].append(skill)
            else:
                missing[category].append(skill)
    return matched, missing

def explain_section_mismatch(section_scores, missing_skills, sections):
    explanations = []
    
    sorted_sections = sorted(section_scores.items(), key=lambda x: x[1])
    
    for section, score in sorted_sections:
        if score < 40:
            if section == "Experience":
                exp_skills = extract_skills(sections.get("Experience", ""))
                all_exp_skills = [s for skills in exp_skills.values() for s in skills]
                if len(all_exp_skills) < 3:
                    explanations.append({
                        "section": section,
                        "score": score,
                        "reason": f"Your Experience section scores only {score}% against the JD. It contains very few technical keywords. Employers scan experience for direct evidence of skills used in real work — not just project names.",
                        "fix": "Rewrite each experience bullet to include specific tools, technologies, and measurable outcomes. Example: Instead of 'worked on data pipeline' write 'built ETL pipeline using Python and Pandas processing 50k records daily'."
                    })
                else:
                    explanations.append({
                        "section": section,
                        "score": score,
                        "reason": f"Your Experience section scores {score}% — the language you use describes your work differently from how the JD describes the same work. This is a terminology mismatch not a skills mismatch.",
                        "fix": "Mirror the exact verbs and nouns from the JD in your experience bullets. If JD says 'developed ML pipelines' and you say 'built models' — change your language to match."
                    })
            
            elif section == "Skills":
                explanations.append({
                    "section": section,
                    "score": score,
                    "reason": f"Your Skills section scores {score}% against the JD. Critical skills required by the employer are either missing or not explicitly listed.",
                    "fix": f"Add these missing skills explicitly if you have any exposure: {', '.join([s for skills in missing_skills.values() for s in skills][:5])}. Even beginner level exposure should be listed with proficiency level."
                })
            
            elif section == "Projects":
                explanations.append({
                    "section": section,
                    "score": score,
                    "reason": f"Your Projects section scores {score}%. Project descriptions are too vague or use different technology stack than what the JD requires.",
                    "fix": "Each project description should mention the specific problem, the tools used, and a quantifiable result. Connect project outcomes to business or research impact."
                })

    if not explanations:
        explanations.append({
            "section": "Overall",
            "score": sum(section_scores.values()) / len(section_scores),
            "reason": "Your resume is reasonably aligned but uses different terminology than the JD in several places.",
            "fix": "Do a keyword audit — paste the JD and highlight every technical term. Check if each one appears in your resume. If not, add it where relevant."
        })
    
    return explanations

def rewrite_weak_sentences(resume_text, jd_text, section_scores):
    rewrites = []
    sentences = [s.strip() for s in resume_text.split('.') if len(s.strip()) > 20]
    
    jd_keywords = extract_skills(jd_text)
    all_jd_keywords = [s for skills in jd_keywords.values() for s in skills]
    
    weak_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        keyword_count = sum(1 for kw in all_jd_keywords if kw in sentence_lower)
        if keyword_count == 0 and len(sentence.split()) > 5:
            weak_sentences.append(sentence)
    
    weak_sentences = weak_sentences[:4]
    
    improvement_patterns = {
        "built": "architected and deployed",
        "worked on": "engineered and optimized",
        "helped": "independently contributed to",
        "did": "executed and delivered",
        "made": "developed and implemented",
        "used": "leveraged",
        "learned": "gained hands-on proficiency in",
        "studied": "conducted in-depth research on",
        "created": "designed and implemented",
        "tested": "validated and quality-assured"
    }
    
    for sentence in weak_sentences:
        rewritten = sentence
        improved = False
        for weak_word, strong_word in improvement_patterns.items():
            if weak_word in rewritten.lower():
                rewritten = re.sub(weak_word, strong_word, rewritten, flags=re.IGNORECASE, count=1)
                improved = True
        
        if improved or len(weak_sentences) < 2:
            rewrites.append({
                "original": sentence,
                "rewritten": rewritten,
                "reason": "This sentence contains no keywords from the job description and uses weak action verbs that do not demonstrate technical depth."
            })
    
    return rewrites

def highlight_jd_sentences(jd_text, resume_skills):
    sentences = [s.strip() for s in jd_text.split('.') if s.strip()]
    all_resume_skills = [s for skills in resume_skills.values() for s in skills]
    highlighted = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        matched_kws = [skill for skill in all_resume_skills if skill in sentence_lower]
        
        if matched_kws:
            highlighted.append({
                "text": sentence + ".",
                "status": "matched",
                "keywords": matched_kws
            })
        else:
            highlighted.append({
                "text": sentence + ".",
                "status": "missing",
                "keywords": []
            })
    
    return highlighted

def create_radar_chart(section_scores):
    categories = list(section_scores.keys())
    values = list(section_scores.values())
    values_normalized = [min(v, 100) for v in values]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_normalized + [values_normalized[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.2)',
        line=dict(color='rgba(102, 126, 234, 0.8)', width=2),
        name='Your Resume'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[100] * (len(categories) + 1),
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(200, 200, 200, 0.1)',
        line=dict(color='rgba(200, 200, 200, 0.5)', width=1, dash='dash'),
        name='Perfect Match'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=400,
        margin=dict(t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_skill_bar_chart(matched, missing):
    categories = list(SKILL_DATABASE.keys())
    matched_counts = [len(matched.get(cat, [])) for cat in categories]
    missing_counts = [len(missing.get(cat, [])) for cat in categories]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Matched Skills', x=categories, y=matched_counts,
                         marker_color='#4CAF50', text=matched_counts, textposition='auto'))
    fig.add_trace(go.Bar(name='Missing Skills', x=categories, y=missing_counts,
                         marker_color='#f44336', text=missing_counts, textposition='auto'))
    fig.update_layout(
        barmode='group',
        height=350,
        xaxis_tickangle=-20,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=80)
    )
    return fig

# ── UI ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Resume Matcher", layout="wide", page_icon="🎯")

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem; font-weight: bold; text-align: center;
    background: linear-gradient(90deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.sub-header { text-align:center; color:#666; margin-bottom:2rem; font-size:1rem; }
.score-box {
    padding: 20px; border-radius: 15px; text-align: center;
    font-size: 3rem; font-weight: bold; margin: 20px 0;
}
.section-score-card {
    background: #f8f9fa; padding: 15px; border-radius: 10px;
    border-left: 4px solid #667eea; margin: 8px 0;
}
.explanation-box {
    background: #fce4ec; padding: 15px; border-radius: 10px;
    border-left: 4px solid #e91e63; margin: 10px 0;
}
.fix-box {
    background: #e8f5e9; padding: 12px; border-radius: 8px;
    border-left: 4px solid #4CAF50; margin-top: 8px; font-size: 0.9rem;
}
.rewrite-original {
    background: #ffebee; padding: 12px; border-radius: 8px;
    font-size: 0.9rem; margin: 5px 0; border-left: 3px solid #f44336;
}
.rewrite-new {
    background: #e8f5e9; padding: 12px; border-radius: 8px;
    font-size: 0.9rem; margin: 5px 0; border-left: 3px solid #4CAF50;
}
.matched-sentence {
    background: #c8e6c9; padding: 6px 10px; border-radius: 5px;
    margin: 4px 0; display: block; font-size: 0.9rem;
}
.missing-sentence {
    background: #ffcdd2; padding: 6px 10px; border-radius: 5px;
    margin: 4px 0; display: block; font-size: 0.9rem;
}
.skill-tag-matched {
    background: #d4edda; padding: 6px 12px; border-radius: 20px;
    display: inline-block; margin: 3px; font-size: 0.82rem; border: 1px solid #28a745;
}
.skill-tag-missing {
    background: #fff3cd; padding: 6px 12px; border-radius: 20px;
    display: inline-block; margin: 3px; font-size: 0.82rem; border: 1px solid #ffc107;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎯 Smart Resume ↔ Job Matcher</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Section-wise analysis · Skill gap detection · Explainability · Sentence rewriter</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📄 Your Resume")
    resume_text = st.text_area("Resume", height=380,
        placeholder="Paste your complete resume here...", label_visibility="collapsed")
with col2:
    st.markdown("### 💼 Job Description")
    jd_text = st.text_area("JD", height=380,
        placeholder="Paste the job description here...", label_visibility="collapsed")

analyze_btn = st.button("🔍 Analyze Match", use_container_width=True, type="primary")

if analyze_btn:
    if not resume_text.strip() or not jd_text.strip():
        st.error("Please paste both resume and job description.")
    else:
        with st.spinner("Running deep analysis..."):
            sections = extract_resume_sections(resume_text)
            section_scores = compute_section_scores(sections, jd_text)
            resume_skills = extract_skills(resume_text)
            jd_skills = extract_skills(jd_text)
            matched_skills, missing_skills = get_skill_gap_analysis(resume_skills, jd_skills)
            
            total_jd = sum(len(s) for s in jd_skills.values())
            total_matched = sum(len(s) for s in matched_skills.values())
            skill_score = (total_matched / total_jd * 100) if total_jd > 0 else 0
            
            valid_scores = [v for v in section_scores.values() if v > 0]
            section_avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
            final_score = (section_avg * 0.5) + (skill_score * 0.5)
            
            explanations = explain_section_mismatch(section_scores, missing_skills, sections)
            rewrites = rewrite_weak_sentences(resume_text, jd_text, section_scores)
            highlighted_jd = highlight_jd_sentences(jd_text, resume_skills)

        st.markdown("---")

        if final_score >= 75:
            color, label, emoji = "#4CAF50", "Strong Match 🚀", "🟢"
        elif final_score >= 50:
            color, label, emoji = "#FF9800", "Moderate Match ⚡", "🟡"
        else:
            color, label, emoji = "#f44336", "Weak Match — Action Needed", "🔴"

        st.markdown(f"""
        <div class="score-box" style="background:linear-gradient(135deg,{color}22,{color}44);border:3px solid {color};">
            {emoji} {final_score:.1f} / 100
            <div style="font-size:1rem;color:{color};margin-top:5px;">{label}</div>
        </div>""", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Section Alignment", f"{section_avg:.1f}%")
        with m2:
            st.metric("Skill Match Rate", f"{skill_score:.1f}%")
        with m3:
            st.metric("Skills Matched", f"{total_matched}")
        with m4:
            st.metric("Skills Missing", f"{sum(len(s) for s in missing_skills.values())}")

        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Section Analysis",
            "🔴 Why Mismatch",
            "❌ Skill Gaps",
            "✏️ Sentence Rewriter",
            "📋 JD Breakdown",
            "📈 Visual Report"
        ])

        with tab1:
            st.markdown("### How Each Resume Section Scores Against the JD")
            st.markdown("This shows which section of your resume is letting you down the most.")
            for section, score in sorted(section_scores.items(), key=lambda x: x[1]):
                if score > 0:
                    bar_color = "#4CAF50" if score >= 60 else "#FF9800" if score >= 35 else "#f44336"
                    st.markdown(f"""
                    <div class="section-score-card">
                        <strong>{section}</strong>
                        <div style="background:#eee;border-radius:10px;margin-top:8px;height:18px;">
                            <div style="background:{bar_color};width:{min(score,100)}%;height:18px;
                            border-radius:10px;text-align:right;padding-right:8px;
                            font-size:0.75rem;line-height:18px;color:white;">
                                {score}%
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="section-score-card" style="opacity:0.5;">
                        <strong>{section}</strong> — Section not detected in resume
                    </div>""", unsafe_allow_html=True)

        with tab2:
            st.markdown("### Root Cause Analysis — Exactly Why You Don't Match")
            for exp in explanations:
                st.markdown(f"""
                <div class="explanation-box">
                    <strong>📍 {exp['section']} Section — {exp['score']:.1f}% match</strong><br><br>
                    🔍 <strong>Why:</strong> {exp['reason']}
                    <div class="fix-box">
                        💡 <strong>How to fix:</strong> {exp['fix']}
                    </div>
                </div>""", unsafe_allow_html=True)

        with tab3:
            st.markdown("### Skill Gap Analysis by Category")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### ❌ Missing Skills")
                for cat, skills in missing_skills.items():
                    if skills:
                        st.markdown(f"**{cat}**")
                        st.markdown("".join([f'<span class="skill-tag-missing">⚠️ {s}</span>' for s in skills]),
                                   unsafe_allow_html=True)
                        st.markdown("")
            with c2:
                st.markdown("#### ✅ Matched Skills")
                for cat, skills in matched_skills.items():
                    if skills:
                        st.markdown(f"**{cat}**")
                        st.markdown("".join([f'<span class="skill-tag-matched">✅ {s}</span>' for s in skills]),
                                   unsafe_allow_html=True)
                        st.markdown("")

        with tab4:
            st.markdown("### ✏️ Weak Sentence Detector and Rewriter")
            st.markdown("These sentences from your resume have zero overlap with the job description. Here is how to fix them:")
            if rewrites:
                for i, rw in enumerate(rewrites, 1):
                    st.markdown(f"**Sentence {i}**")
                    st.markdown(f'<div class="rewrite-original">❌ <strong>Original:</strong> {rw["original"]}</div>',
                               unsafe_allow_html=True)
                    st.markdown(f'<div class="rewrite-new">✅ <strong>Stronger version:</strong> {rw["rewritten"]}</div>',
                               unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:0.8rem;color:#888;margin-bottom:15px;">💬 {rw["reason"]}</div>',
                               unsafe_allow_html=True)
            else:
                st.success("All your sentences contain relevant keywords. Focus on quantifying your achievements.")

        with tab5:
            st.markdown("### JD Sentence Level Breakdown")
            st.markdown("🟢 Your resume covers this | 🔴 Gap identified")
            for item in highlighted_jd:
                if item["status"] == "matched":
                    kw_str = ", ".join(item["keywords"][:3])
                    st.markdown(f'<span class="matched-sentence">✅ {item["text"]} <em style="font-size:0.75rem;color:#2e7d32;">({kw_str})</em></span>',
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="missing-sentence">❌ {item["text"]}</span>',
                               unsafe_allow_html=True)

        with tab6:
            st.markdown("### Visual Analysis Report")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Section Score Radar")
                valid_section_scores = {k: v for k, v in section_scores.items() if v > 0}
                if len(valid_section_scores) >= 3:
                    st.plotly_chart(create_radar_chart(valid_section_scores), use_container_width=True)
                else:
                    st.info("Need at least 3 detected sections for radar chart.")
            with c2:
                st.markdown("#### Skill Match vs Gap by Category")
                st.plotly_chart(create_skill_bar_chart(matched_skills, missing_skills), use_container_width=True)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#999;font-size:0.8rem;">Smart Resume Matcher — Section-wise NLP Analysis with Explainability</div>',
           unsafe_allow_html=True)