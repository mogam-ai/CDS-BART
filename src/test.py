from huggingface_hub import create_repo, login
from transformers import BartModel, BartTokenizerFast

# login()

import os

token = os.environ.get("")HF_TOKEN

repo_id = "mogam-ai/CDS-BART-denoising"
model = BartModel.from_pretrained(repo_id)
tokenizer = BartTokenizerFast.from_pretrained(repo_id)

print(model)
print(tokenizer)


fake_id = "mogam-ai/fake"
create_repo(
    repo_id,
    exist_ok=True,
    repo_type="model",
    private=True,
    resource_group_id="683695bbaffae1c74f417a89",
)
print(f"Created repository {fake_id} successfully.")
