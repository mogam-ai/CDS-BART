# %%
from datasets import load_from_disk
from functools import partial
import os
import torch, os, sys, shutil
# # from torch.utils.data import DataLoader
from transformers import (
    BartModel,
    BartTokenizer,
    DataCollatorWithPadding,
)

import pandas as pd
from pathlib import Path

# 환경 setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tokenizer_path = Path('/fsx/s3/project/P240002_mRNA_LLM/rna_data/tokens/rna_spmbpe_60M/BartSPBPETokenizerFast')
best_model = Path('/fsx/s3/project/P240002_mRNA_LLM/checkpoints/finetune/Ecoli_classification_1/checkpoint-2040')
# best_model = Path("/fsx/s3/project/P240002_mRNA_LLM/checkpoints/pretrain/bart_seq2seq_noise_850_large_last/checkpoint-316395")

out_dir = './results'

# output directory 만들기
out_dir = Path(out_dir)
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True)
print(f'Output directory created at {out_dir}')

# load the model and the tokenizer
model = BartModel.from_pretrained(best_model)
tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
data_collator = DataCollatorWithPadding(tokenizer)

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
def load_the_dataset(data_path,data_name):
    data = load_from_disk(data_path)
    ds = data['train']

    seq_list = ds['sequence']
    label_list = ds['y']
    length_list = [len(seq) for seq in seq_list]

    df = pd.DataFrame({'sequence':seq_list, 'y':label_list, 'len':length_list})
    df.to_csv(out_dir.joinpath(f"{data_name}_df.csv"), index=False)

    model.to("cuda")

    # eos token 추출
    list_emb_with_eos = []
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
            
            list_emb_with_eos.append(output_embeds[-1].unsqueeze(0))   # [1, 1, 768]
            list_emb_no_eos.append(output_embeds[1:-1])
    
    emb_with_eos = torch.stack(list_emb_with_eos).to("cpu")  # [1500, 1, 768]
    emb_no_eos = torch.stack(list_emb_no_eos).to("cpu")
    print(f"Shape of embeddings with eos: {emb_with_eos.shape}")
    print(f"Shape of embeddings without eos: {emb_no_eos.shape}")

    with open(out_dir.joinpath(f"emb_with_eos_{data_name}.pt"), "wb") as f:
        torch.save(emb_with_eos, f)

    with open(out_dir.joinpath(f"emb_no_eos_{data_name}.pt"), "wb") as f:
        torch.save(emb_no_eos, f)

# load e.coli train huggingface dataset 
data_path='/fsx/home/jhhong/mogam_project/CDS-BART-project/supplementary/Supplementary/Embedding_space_data/finetune_ecoli'
data_name = 'ecoli'

load_the_dataset(data_path, data_name)