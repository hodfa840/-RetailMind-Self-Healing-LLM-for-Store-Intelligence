import os
from huggingface_hub import HfApi

token = open("hf_token").read().strip()
api = HfApi(token=token)

repo_id = "Hodfa71/RetailMind"
print(f"Creating Space: {repo_id}")
api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)

print("Uploading files...")
api.upload_folder(
    folder_path=".",
    repo_id=repo_id,
    repo_type="space",
    allow_patterns=["*.py", "*.txt", "*.md", "modules/**/*.py", "tests/**/*.py"]
)
print(f"Deployed to https://huggingface.co/spaces/{repo_id}")
