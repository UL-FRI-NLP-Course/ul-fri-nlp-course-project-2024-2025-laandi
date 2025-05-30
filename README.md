# Natural language processing course: `Project 1: Conversational Agent with Retrieval-Augmented Generation`

## Smart Academic Research Assistant: Conversational Agent with Retrieval-Augmented Generation

This repository contains our research project for the Natural Language Processing course at the Faculty of Computer and Information Science, University of Ljubljana (2024/2025).

## Project Overview

We present a specialized conversational agent designed to enhance academic research by combining Retrieval-Augmented Generation (RAG) with web scraping techniques. The system aims to provide accurate, citation-backed responses while addressing common limitations of traditional language models such as outdated information and hallucinations.

### Research Goals

- Develop a more focused and precise academic research assistant, specifically in the field of Natural Language Processing
- Integrate multiple academic paper sources through Python libraries, API interfaces and web scraping
- Implement efficient vector similarity search using FAISS
- Evaluate the effectiveness of combining RAG with real-time web data

### Project Structure

```
.
├── code                    # Main implementation code
    ├── old/                # Legacy files - baseline implementation, testing, database construction
├── report/                 # LaTeX report and documentation
    ├── code/               # Evaluation code and test scripts
    └── evaluations_figs/   # Evaluation figures
    └── fig/                # Other figures and resources used in the report
```

### Instructions for running the code
1. Move to our **working repository on HPC** with: `cd /d/hpc/projects/onj_fri/laandi/ul-fri-nlp-course-project-2024-2025-laandi/code`.
2. To **test** our agent, run `./baseline_script.sh`. As this bash script already contains the `srun` command, you do not have to worry about anything else.
3. The script will first ask you to enter **keywords**. These are optional, but if your question in the next step is vague, it is better to provide keywords for improved results. Press enter if you want to skip.
4. Pose your **question** to the model. As you know, these models can be sensitive to poorly phrased prompts, so try to make it as concise and clear as possible.
5. The model will then ask whether you want to retrieve **new papers**. Keep in mind that the current database consists of articles from the natural language processing field. If your question is from a different domain, choose "y". If you are just testing the model, choose "n".

**Non-working nodes problem**

In light of recent issues with some HPC nodes being unavailable, you are allowed to modify the following line in `baseline_script.sh`:

`srun --gres=gpu:1 --partition=gpu singularity exec --nv  /d/hpc/projects/onj_fri/laandi/containers/container_llm.sif python3 -c "`

to something like:

`srun --gres=gpu:1 --partition=gpu --exclude=wn224 singularity exec --nv  /d/hpc/projects/onj_fri/laandi/containers/container_llm.sif python3 -c "`,

where **wn224** is the node you want to exclude. Or just try your luck :) 

### Database restoration
If at any point something goes wrong and the database is deleted or corrupted, please restore it using the following commands:
```
cd /d/hpc/projects/onj_fri/laandi/ul-fri-nlp-course-project-2024-2025-laandi/code
srun singularity exec /d/hpc/projects/onj_fri/laandi/containers/container_llm.sif python3 restore_db.py
```

## Team

- Anja Pijak
- Lara Anžur
- Dimitrije Stefanović

### Advisor

- Aleš Žagar
