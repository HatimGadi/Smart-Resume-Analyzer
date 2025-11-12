# 🧠 Smart Resume Analyzer

> **AI-powered Resume Evaluator, ATS Builder & Skill Analyzer using Google Gemini and NLP**

Smart Resume Analyzer is an AI-driven web app built with **Streamlit** that evaluates resumes, checks **ATS compatibility**, and suggests improvements using **Natural Language Processing (NLP)** and **Google Gemini AI**.  
It bridges the gap between HR and Data Science, helping users create optimized, job-ready resumes.

---

## 🚀 Features

- ✅ **Skill Analyzer** – Detects matched and missing skills from your resume using fuzzy matching and trending skill data.  
- ✅ **ATS Resume Builder** – Builds ATS-friendly resumes with clean, modern templates.  
- ✅ **AI Suggestions** – Gemini provides targeted recommendations for missing skills or weak sections.  
- ✅ **Live Skill Updates** – Fetches trending skills per domain (Web Dev, Data Science, ML, etc.).  
- ✅ **AI Experience Enhancer** – Rewrites your experience section to sound more professional.  

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **AI & NLP** | Google Gemini API |
| **Matching Engine** | RapidFuzz |
| **Data Handling** | Pandas |
| **Env Management** | python-dotenv |
| **Language** | Python 3.10+ |

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Smart-Resume-Analyzer.git
cd Smart-Resume-Analyzer
```
---

#### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate  # On Mac/Linux
```
---

### 3. Install dependencies
```bash

pip install -r requirements.txt
```
---

### 🔑 Environment Setup
```bash
Create a .env file in your project root and add your Gemini API key:
GOOGLE_API_KEY=your_actual_gemini_api_key
```
---
### ▶️ Run the App
```bash
streamlit run resume.py
```
---

### 📊 Project Modules

| File | Description |
|------------|-------------|
| **resume.py** | Main Streamlit app (Skill Analyzer + Resume Builder) |
| **utils.py** | Resume parsing and PDF rendering utilities |
| **web_scraper.py** | Fetches trending domain-specific skills dynamically|

🧠 How It Works
- **Upload Resume** → Extracts text and identifies skills using fuzzy matching.
- **Fetch Trending Skills** → Uses a scraper to collect the latest domain-specific skills.
- **Analyze Skills** → Calculates ATS score based on matched vs missing skills.
- **AI Feedback (Gemini)** → Suggests improvements for missing skills or weak experience lines.
- **Build Resume** → Generates an ATS-friendly, downloadable resume using selected templates.

### 📈 Example Output

- ATS Score: 84% 
- Matched Skills: Python, Flask, SQL, APIs
- Missing Skills: Kubernetes, Docker

***💡 AI Suggestion (Gemini):***

Add a section highlighting deployment experience with Docker or Kubernetes.
Include measurable project outcomes to strengthen your experience section.

### 🧩 Future Enhancements
- 🔹 Add Job Description (JD) fit analysis module
- 🔹 Improve ATS scoring by analyzing layout and formatting
- 🔹 LinkedIn profile import and auto-analysis
- 🔹 Resume comparison dashboard and visualization
- 🔹 FastAPI backend for scalable deployment