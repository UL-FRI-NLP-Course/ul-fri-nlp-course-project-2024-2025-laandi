import matplotlib.pyplot as plt
import numpy as np
import os

data = {
    'Question 1': {
        'Baseline + VectorStore': [9, 6, 3],
        'VectorStore + Fetch': [7, 5, 5],
        'ChatGPT': [4, 6, 7],
    },
    'Question 2': {
        'Baseline + VectorStore': [2, 7, 9],
        'VectorStore + Fetch': [10, 6, 2],
        'ChatGPT': [9, 4, 6],
    },
    'Question 3': {
        'Baseline + VectorStore': [13, 4, 1],
        'VectorStore + Fetch': [3, 8, 7],
        'ChatGPT': [3, 6, 9],
    },
    'Question 4': {
        'Baseline + VectorStore': [11, 5, 2],
        'VectorStore + Fetch': [4, 6, 8],
        'ChatGPT': [4, 7, 7],
    },
    'Question 5': {
        'Baseline + VectorStore': [5, 6, 7],
        'VectorStore + Fetch': [10, 5, 3],
        'ChatGPT': [5, 6, 7],
    },
    'Question 6': {
        'Baseline + VectorStore': [3, 5, 10],
        'VectorStore + Fetch': [10, 6, 2],
        'ChatGPT': [5, 7, 6],
    }
}
output_folder = r'C:\Users\EC\Desktop\master1\nlp\ul-fri-nlp-course-project-2024-2025-laandi\report\evaluation_figs'

for question, answers in data.items():
    answer_labels = list(answers.keys())
    
    # Convert counts to percentages per model
    percentages = {
        model: (np.array(ranks) / sum(ranks)) * 100 if sum(ranks) > 0 else np.array([0, 0, 0])
        for model, ranks in answers.items()
    }

    rank_1 = np.array([percentages[model][0] for model in answer_labels])
    rank_2 = np.array([percentages[model][1] for model in answer_labels])
    rank_3 = np.array([percentages[model][2] for model in answer_labels])
    y_pos = np.arange(len(answer_labels))

    plt.figure(figsize=(8, 5))
    plt.barh(y_pos, rank_1, color='blue', label='Rank 1')
    plt.barh(y_pos, rank_2, left=rank_1, color='orange', label='Rank 2')
    plt.barh(y_pos, rank_3, left=rank_1 + rank_2, color='green', label='Rank 3')

    # Add percentages text
    for i in range(len(answer_labels)):
        plt.text(rank_1[i]/2, y_pos[i], f'{rank_1[i]:.1f}%', va='center', ha='center', color='white', fontsize=9)
        plt.text(rank_1[i] + rank_2[i]/2, y_pos[i], f'{rank_2[i]:.1f}%', va='center', ha='center', color='white', fontsize=9)
        plt.text(rank_1[i] + rank_2[i] + rank_3[i]/2, y_pos[i], f'{rank_3[i]:.1f}%', va='center', ha='center', color='white', fontsize=9)

    plt.yticks(y_pos, answer_labels)
    plt.xlabel('Percentage of Responses')
    plt.title(question)
    plt.legend(loc='upper right')
    plt.tight_layout()
    filename = os.path.join(output_folder, f"{question.replace(' ', '_').lower()}.svg")
    plt.savefig(filename, format='svg')
    
