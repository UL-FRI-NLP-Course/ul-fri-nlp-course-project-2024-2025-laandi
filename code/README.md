## Instructions for running the code
1. Move to the directory `/d/hpc/projects/onj_fri/laandi`.
2. Allocate resources by calling the command `srun --gres=gpu:1 --partition=gpu --time=1:00:00 --pty singularity exec --nv --bind /d/hpc/projects/onj_fri/laandi:/workspace /d/hpc/projects/onj_fri/laandi/containers/container_llm.sif bash`. This will allow you to work in terminal inside the container. You can customize time for allocation of resources.

### Loading new models
To load new model from HuggingFace, use the script `load_model.py`.

### Building database
To add new articles to the database, run `python3 build_database.py`. The sources, number of articles per source and chunk size for keywords are customizable in the code.
`TODO:` arg parser

### Using retriever on database
`TODO`: adapt code from baseline.py