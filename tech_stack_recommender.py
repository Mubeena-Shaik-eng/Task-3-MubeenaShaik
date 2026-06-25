"""
Project 3: AI Recommendation Logic
Tech Stack Recommender using TF-IDF and Cosine Similarity
DecodeLabs AI Internship - Batch 2026
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load dataset (job roles = "items" in our recommendation engine)
data = pd.read_csv("raw_skills.csv")

# 2. Build TF-IDF vectors from each job role's skill set
vectorizer = TfidfVectorizer()
job_vectors = vectorizer.fit_transform(data["skills"])

# 3. Take user input (minimum 3 skills, as required)
user_skills = input("Enter your skills (comma-separated, e.g. Python, Cloud, Automation): ")
user_skills_list = [skill.strip() for skill in user_skills.split(",")]

if len(user_skills_list) < 3:
    print("Please enter at least 3 skills for accurate matching.")
else:
    # 4. Transform user input into the SAME vector space as job roles
    user_text = " ".join(user_skills_list)
    user_vector = vectorizer.transform([user_text])

    # 5. Score: cosine similarity between user vector and every job vector
    scores = cosine_similarity(user_vector, job_vectors).flatten()

    # 6. Sort and filter: rank job roles, drop zero-relevance matches, take Top 3
    data["match_score"] = scores
    top_matches = data[data["match_score"] > 0].sort_values(by="match_score", ascending=False).head(3)

    # 7. Output
    print(f"\nBased on your skills: {user_skills_list}")
    if top_matches.empty:
        print("No relevant matches found. Try different or more common skill keywords.")
    else:
        print(f"Top {len(top_matches)} recommended career path(s):\n")
        for rank, row in enumerate(top_matches.itertuples(), start=1):
            match_percent = round(row.match_score * 100, 1)
            print(f"{rank}. {row.job_role}  —  {match_percent}% match")
