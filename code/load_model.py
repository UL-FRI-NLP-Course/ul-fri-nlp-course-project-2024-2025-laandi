from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    #allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"],
    cache_dir="../../models/hf-models",  # or another persistent folder
    local_dir="../../models/mistral-7b-v3",  # where to store the final files
    local_dir_use_symlinks=False  # copy files instead of symlinks
)

print(f"Model downloaded to: {local_path}")
