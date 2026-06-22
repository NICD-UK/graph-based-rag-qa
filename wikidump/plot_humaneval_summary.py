#!/usr/bin/env python3
"""Plot humaneval summary as a bar chart."""

import csv
import matplotlib.pyplot as plt

# Read data
dump_tags = []
answerable = []
total = []

with open("humaneval_summary.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        dump_tags.append(row["Dump tag"])
        answerable.append(int(row["Answerable questions"]))
        total.append(int(row["Total questions"]))

# Calculate percentages
pct_answerable = [a / t * 100 for a, t in zip(answerable, total)]

# Create figure with two y-axes
fig, ax1 = plt.subplots(figsize=(6, 5))
ax2 = ax1.twinx()

# Bar chart for answerable questions
bars = ax1.bar(dump_tags, answerable, color="steelblue", alpha=0.8, width=0.6)
ax1.set_xlabel("Dump Tag")
ax1.set_ylabel("Answerable Questions")
total_questions = total[0]  # All totals are the same (1315)
ax1.set_ylim(1100, total_questions)

# Right axis shows percentage, aligned to the same data
ax2.set_ylabel("% Answerable Questions")
ax2.set_ylim(1100 / total_questions * 100, 100)

# Rotate x labels for readability
ax1.tick_params(axis="x", rotation=90)

plt.title("Answerable Questions by Wikipedia Dump")
plt.tight_layout()
plt.savefig("humaneval_summary.png", dpi=150)
print("Saved: humaneval_summary.png")
