"""
generate_dataset.py
-------------------
Generates a realistic synthetic dataset for student exam scores based on:
1. Hours spent studying per week (feature 1)
2. Number of practice questions completed (feature 2)
The target is the Exam Score (0 to 100).
"""

import numpy as np

def generate_data(num_samples: int = 150, seed: int = 42):
    # Set a random seed so the data generated is identical every time you run it
    np.random.seed(seed)

    # 1. Generate realistic study hours (between 1.0 and 10.0 hours per week)
    study_hours = np.random.uniform(1.0, 10.0, size=(num_samples, 1))

    # 2. Generate practice questions completed (between 10 and 100 questions)
    practice_questions = np.random.uniform(10.0, 100.0, size=(num_samples, 1))

    # 3. Define the ground truth relationship (the hidden law of nature):
    # Score = (study_hours * 4.5) + (practice_questions * 0.35) + 30 (base score) + noise
    noise = np.random.normal(loc=0.0, scale=3.5, size=(num_samples, 1))

    exam_scores = (study_hours * 4.5) + (practice_questions * 0.35) + 30.0 + noise

    # Clip scores so they stay realistically between 0 and 100
    exam_scores = np.clip(exam_scores, 0.0, 100.0)

    # Combine into a single 2D array: [study_hours, practice_questions, exam_score]
    data = np.hstack([study_hours, practice_questions, exam_scores])

    # Round to 2 decimal places for clean display
    data = np.round(data, 2)

    # Save to CSV
    header = "study_hours,practice_questions,exam_score"
    np.savetxt("data/student_scores.csv", data, delimiter=",", header=header, comments="", fmt="%.2f")
    print(f"Dataset generated successfully with {num_samples} samples -> data/student_scores.csv")

if __name__ == "__main__":
    generate_data()

