# Project 3 — Tech Stack Recommender (AI Recommendation Logic)

DecodeLabs AI Internship | Batch 2026

## Overview
A content-based recommendation engine that maps a user's skills to the most relevant career/job roles, using **TF-IDF vectorization** and **Cosine Similarity** — the same core math behind real-world systems like Netflix and Amazon recommendations.

## Pipeline (Input → Process → Output)
1. **Ingestion** — `raw_skills.csv` stores 20 job roles, each tagged with its typical skill set
2. **Vectorization** — `TfidfVectorizer` converts skill-text into weighted numerical vectors, so rare/specific skills (e.g. "Kubernetes") carry more weight than generic ones
3. **User Input** — the user enters at least 3 skills via the terminal
4. **Scoring** — the user's skill vector is compared against every job role vector using cosine similarity (angle-based, so it isn't biased by list length)
5. **Sorting & Filtering** — roles are ranked by similarity score, zero-relevance matches are dropped, and the Top 3 are shown

## Tech Stack
- Python 3
- pandas
- scikit-learn (`TfidfVectorizer`, `cosine_similarity`)

## How to Run
```bash
pip install pandas scikit-learn --break-system-packages
python tech_stack_recommender.py
```
Then enter skills when prompted, e.g.:
```
Enter your skills (comma-separated, e.g. Python, Cloud, Automation): Python, Machine Learning, Statistics
```

## Sample Output
```
Top 3 recommended career path(s):

1. Data Scientist        — 63.0% match
2. AI Research Engineer  — 60.0% match
3. Machine Learning Engineer — 58.9% match
```

## Why TF-IDF + Cosine Similarity (not just keyword overlap)
- Simple binary tag-matching treats every skill equally, which fails — a generic skill like "Python" shouldn't carry the same weight as a specific one like "Kubernetes"
- **TF-IDF** automatically down-weights common terms and up-weights distinctive ones
- **Cosine similarity** measures the *angle* between vectors instead of raw distance, so it isn't thrown off by users who list more or fewer skills than a job role's tag count

## Known Limitation (Cold Start)
This is content-based filtering, so it has no "history" — it can recommend brand-new job roles instantly (no item cold start), but a user with completely unrecognized/misspelled skills will get no matches, since their vector has nothing to align with.

## Author
Mubeena — AI Intern, DecodeLabs (Batch 2026)
