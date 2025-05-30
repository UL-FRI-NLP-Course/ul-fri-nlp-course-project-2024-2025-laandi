#!/bin/bash

# Prompt for keywords
read -p "Enter keywords to search for papers: " keywords
if [[ -z "$keywords" ]]; then
    echo "Keywords cannot be empty."
    exit 1
fi

# Prompt for maximum number of documents per source
read -p "Enter maximum number of documents per source (default: 5): " max_docs
if [[ -z "$max_docs" ]]; then
    max_docs=5
elif ! [[ "$max_docs" =~ ^[0-9]+$ ]]; then
    echo "Maximum documents must be a number."
    exit 1
fi

echo "Fetching up to $max_docs papers per source related to '$keywords'..."

# Run srun with the SIF image and call the Python function
srun --gres=gpu:1 --partition=gpu singularity exec --nv ../../containers/container_llm.sif python3 -c "
from code.fetch_papers import fetch_and_save_papers
vectorstore = fetch_and_save_papers(
    keywords='$keywords',
    max_docs=$max_docs
)
print('Papers successfully fetched and saved to vector store.')
"

# Make the script executable after creating it
# chmod +x fetch_papers.sh
