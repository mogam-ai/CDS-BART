# %%
from datasets import load_from_disk
from functools import partial
import os
# import torch, os, sys
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
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# path of tokenizer and ckp model
refseq = Path('/fsx/s3/project/P240002_mRNA_LLM/raw_data/genomes_refseq/')
tokenizer_path = Path('/fsx/s3/project/P240002_mRNA_LLM/rna_data/tokens/rna_spmbpe_60M/BartSPBPETokenizerFast')
ckp_model2 = Path("/fsx/s3/project/P240002_mRNA_LLM/checkpoints/pretrain/bart_seq2seq_noise_850_large_last/checkpoint-316395")
outdir = Path('Embedding_space_data')

# data_name = 'CDSBART_BEST'
# data_path = Path(f"/fsx/home/jhhong/2024_project/2.RNA_LLM/Supplementary/Embedding_space_data/{data_name}")
# data_path = Path("/fsx/home/jhhong/2024_project/2.RNA_LLM/Supplementary/Embedding_space_data/CDSBART_ALL")
# data_path = Path('/fsx/home/jhhong/2024_project/2.RNA_LLM/Supplementary/Embedding_space_data/CODONBERT_ALL_SPEICES')

# check exist with assertion
# assert tokenizer_path.is_dir(), f'Tokenizer not found at {tokenizer_path}'
# assert refseq.is_dir(), f'Checkpoint model 2 not found in {refseq}'

out_dir = './results'

# output directory 만들기
out_dir = Path(out_dir)
if not out_dir.exists():
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory created at {out_dir}')

# load the model and the tokenizer
model = BartModel.from_pretrained(ckp_model2)
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

#%%
# load the dataset
def load_the_dataset(data_path,i):
    data = load_from_disk(data_path)
    ds = data['train']

    seq_list = ds['sequence']
    label_list = ds['y']
    length_list = [len(seq) for seq in seq_list]

    df = pd.DataFrame({'sequence':seq_list, 'y':label_list, 'len':length_list})
    # df.to_csv(out_dir.joinpath('data_codon_bert.csv'), index=False)
    df.to_csv(out_dir.joinpath(f"{data_name}_{i}_df.csv"), index=False)
#%%
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
            
            list_emb_with_eos.append(output_embeds)
            list_emb_no_eos.append(output_embeds[1:-1])
            
    emb_with_eos = torch.stack(list_emb_with_eos)
    emb_no_eos = torch.stack(list_emb_no_eos).to("cpu")
    print(f"Shape of embeddings with eos: {emb_with_eos.shape}")
    print(f"Shape of embeddings without eos: {emb_no_eos.shape}")

    with open(out_dir.joinpath(f"emb_with_eos_{data_name}_{i}.pt"), "wb") as f:
        torch.save(emb_with_eos, f)

    with open(out_dir.joinpath(f"emb_no_eos_{data_name}_{i}.pt"), "wb") as f:
        torch.save(emb_no_eos, f)
        
# for i in range(8,10):
#     data_name = 'CDSBART_BEST'
#     data_path = Path(f"/fsx/home/jhhong/2024_project/2.RNA_LLM/Supplementary/Embedding_space_data/{data_name}_{i}")
#     load_the_dataset(data_path,str(i))

data_name = 'CDSBART_ALL_SPECIES_final'
data_path = Path(f"/fsx/home/jhhong/2024_project/2.RNA_LLM/Supplementary/Embedding_space_data/{data_name}")
load_the_dataset(data_path,0)