import matplotlib.pyplot as plt
import numpy as np

data = {
    'Question 2': {
        'Baseline + VectorStore': [9, 6, 3],
        'VectorStore + Fetch': [7, 5, 5],
        'ChatGPT': [4, 6, 7],
    },
    'Question 3': {
        'Baseline + VectorStore': [2, 7, 9],
        'VectorStore + Fetch': [10, 6, 2],
        'ChatGPT': [9, 4, 6],
    },
    'Question 4': {
        'Baseline + VectorStore': [13, 4, 1],
        'VectorStore + Fetch': [3, 8, 7],
        'ChatGPT': [3, 6, 9],
    },
    'Question 5': {
        'Baseline + VectorStore': [11, 5, 2],
        'VectorStore + Fetch': [4, 6, 8],
        'ChatGPT': [4, 7, 7],
    },
    'Question 6': {
        'Baseline + VectorStore': [5, 6, 7],
        'VectorStore + Fetch': [10, 5, 3],
        'ChatGPT': [5, 6, 7],
    },
    'Question 7': {
        'Baseline + VectorStore': [3, 5, 10],
        'VectorStore + Fetch': [10, 6, 2],
        'ChatGPT': [5, 7, 6],
    }
}

# Sum counts for each answer across all questions
summary_counts = {
    'Baseline + VectorStore': np.array([0, 0, 0]),
    'VectorStore + Fetch': np.array([0, 0, 0]),
    'ChatGPT': np.array([0, 0, 0]),
}

for answers in data.values():
    for answer, counts in answers.items():
        summary_counts[answer] += np.array(counts)

# Convert counts to percentages per answer
summary_percentages = {}
for answer, counts in summary_counts.items():
    total = counts.sum()
    summary_percentages[answer] = (counts / total) * 100 if total > 0 else counts

# Prepare data for plotting
answer_labels = list(summary_percentages.keys())
rank_1 = np.array([summary_percentages[ans][0] for ans in answer_labels])
rank_2 = np.array([summary_percentages[ans][1] for ans in answer_labels])
rank_3 = np.array([summary_percentages[ans][2] for ans in answer_labels])
y_pos = np.arange(len(answer_labels))

plt.figure(figsize=(8, 5))
plt.barh(y_pos, rank_1, color='#003f5c', label='Rank 1', height=0.5)
plt.barh(y_pos, rank_2, left=rank_1, color='#ffa600', label='Rank 2', height=0.5)
plt.barh(y_pos, rank_3, left=rank_1 + rank_2, color='#58508d', label='Rank 3', height=0.5)

# Add percentage text on bars
for i in range(len(answer_labels)):
    plt.text(rank_1[i] / 2, y_pos[i], f'{rank_1[i]:.1f}%', va='center', ha='center', color='white', fontsize=10)
    plt.text(rank_1[i] + rank_2[i] / 2, y_pos[i], f'{rank_2[i]:.1f}%', va='center', ha='center', color='white', fontsize=10)
    plt.text(rank_1[i] + rank_2[i] + rank_3[i] / 2, y_pos[i], f'{rank_3[i]:.1f}%', va='center', ha='center', color='white', fontsize=10)

plt.yticks(y_pos, answer_labels,)
plt.xlabel('Percentage of Times Ranked')
plt.title('Summary of Answer Rankings for All Three Models (in %)')
plt.legend()
plt.tight_layout()
plt.savefig('C:/Users/EC/Desktop/master1/nlp/ul-fri-nlp-course-project-2024-2025-laandi/report/evaluation_figs/summary_plot.svg', format='svg')
plt.show()
