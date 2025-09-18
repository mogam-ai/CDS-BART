
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login

import torch, os, sys
from transformers import (
    BartModel,
    BartTokenizer,
    DataCollatorWithPadding,
)

# 환경 setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# output directory 만들기
out_dir = Path('./results')
if not out_dir.exists():
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory created at {out_dir}')

# load the model and the tokenizer
repo_id = "mogam-ai/CDS-BART-denoising"
model = BartModel.from_pretrained(repo_id)
tokenizer = BartTokenizer.from_pretrained(repo_id)
data_collator = DataCollatorWithPadding(tokenizer)
data_repo_id = "mogam-ai/taxonomy-embeddings"


print("Before changing the tokenizer indices")
print(f"BOS token: {tokenizer.bos_token_id}")
print(f"EOS token: {tokenizer.eos_token_id}")
print(f"UNK token: {tokenizer.unk_token_id}")
print(f"SEP token: {tokenizer.sep_token_id}")
print(f"PAD token: {tokenizer.pad_token_id}")
print(f"CLS token: {tokenizer.cls_token_id}")
print(f"MASK token: {tokenizer.mask_token_id}")

tokenizer.bos_token_id = 0
tokenizer.eos_token_id = 2
tokenizer.unk_token_id = 3
tokenizer.sep_token_id = 2
tokenizer.pad_token_id = 1
tokenizer.cls_token_id = 0

print("After changing the tokenizer indices")
print(f"BOS token: {tokenizer.bos_token_id}")
print(f"EOS token: {tokenizer.eos_token_id}")
print(f"UNK token: {tokenizer.unk_token_id}")
print(f"SEP token: {tokenizer.sep_token_id}")
print(f"PAD token: {tokenizer.pad_token_id}")
print(f"CLS token: {tokenizer.cls_token_id}")
print(f"MASK token: {tokenizer.mask_token_id}")

# load the dataset
def load_the_dataset(data_name):
    data = load_dataset(
         data_repo_id, 
        data_files="CDSBART_BEST_3_df.csv")
    
    # data = load_from_disk(data_path)
    ds = data['train']
    seq_list = ds['sequence']
    label_list = ds['y']
    length_list = [len(seq) for seq in seq_list]
    
    df = pd.DataFrame({'sequence':seq_list, 'y':label_list, 'len':length_list})
    df.to_csv(out_dir.joinpath(f"{data_name}_df.csv"), index=False)
    
    model.to("cuda")

    # eos token 추출
    list_emb_no_eos = []
    
    for seq in seq_list: 
        encoded = tokenizer(
        seq, 
        max_length=850, # we need to stack the tensors so need to fix the length, if work on batch level with dynamic padding then  no matter
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
        return_tensors="pt"
        )
        
        # model output
        with torch.no_grad():
            output = model(
                input_ids = encoded['input_ids'].cuda(),
                attention_mask = encoded['attention_mask'].cuda()
            )
        
            hidden_states = output.last_hidden_state
            output_embeds = torch.squeeze(hidden_states[-1])
            print(f'hidden state shape: {hidden_states.shape}')
            print(f'output embedds: {output_embeds.shape}')
            print(f'part tokenizer embedss: {output_embeds[1:-1].shape}')
            
            list_emb_no_eos.append(output_embeds[1:-1])
            
    emb_no_eos = torch.stack(list_emb_no_eos).to("cpu")
    print(f"Shape of embeddings without eos: {emb_no_eos.shape}")

    with open(out_dir.joinpath(f"emb_no_eos_{data_name}.pt"), "wb") as f:
        torch.save(emb_no_eos, f)
        
if __name__ == "__main__":
    data_name = 'taxonomy'
    load_the_dataset(data_name)