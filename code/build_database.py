from fetch_papers import fetch_and_save_papers
import json

with open("nlp_tasks.json", "r") as f:
    nlp_keywords = json.load(f)

# Query every keyword on its own
for i in range(len(nlp_keywords)):
    
    fetch_and_save_papers(
        keywords=nlp_keywords[i],
        max_docs=50
    )


# # Or query chunks of keywords
# for i in range(0, len(nlp_keywords), 10):
#     chunk = " OR ".join([nlp_keywords[j] for j in range(i, min(i+10, len(nlp_keywords)))])
    
#     fetch_and_save_papers(
#         keywords=chunk,
#         max_docs=10
#     )

print('Papers successfully fetched and saved to vector store.')