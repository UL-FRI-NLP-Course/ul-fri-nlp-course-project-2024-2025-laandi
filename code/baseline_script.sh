#!/bin/bash

# Prompt for keywords (optional)
read -p "Enter keywords (optional, press Enter to skip): " keywords
if [[ -z "$keywords" ]]; then
    keywords=None
fi

# Prompt for question (must be a string)
read -p "Enter your question: " question
if [[ -z "$question" ]]; then
    echo "Question cannot be empty."
    exit 1
fi

# Prompt for searching new papers online (yes/no)
read -p "Search for new papers online? (y/n): " search_new
if [[ "$search_new" =~ ^[Yy]$ ]]; then
    search_new=True
else
    search_new=False
fi


#srun --gres=gpu:1 --partition=gpu singularity exec --nv ./containers/container_llm.sif python3 test.py

# Run srun with the SIF image and call the Python function
srun --gres=gpu:1 --partition=gpu singularity exec --nv  /d/hpc/projects/onj_fri/laandi/containers/container_llm.sif python3 -c "

from baseline import ask_dynamic_rag
ask_dynamic_rag(
    question=\"\"\"$question\"\"\",
    keywords=$([[ $keywords == None ]] && echo None || echo \"'$keywords'\"),
    searchNewPaper=$search_new
)
"

# Make the script executable after creating it
# chmod +x baseline_script.sh
