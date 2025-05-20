from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="sentence-transformers/all-mpnet-base-v2",
    cache_dir="../../models/hf-models",  # or another persistent folder
    local_dir="../../models/sentence-transformer",  # where to store the final files
    local_dir_use_symlinks=False  # copy files instead of symlinks
)

print(f"Model downloaded to: {local_path}")
