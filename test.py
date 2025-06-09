from datasets import load_dataset
from transformers import BartModel

auth_token = "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"

names = [
    # "CDS-BART-fungal-expression",
    # "CDS-BART-mRFP-expression",
    # "CDS-BART-SARS-CoV-2-vaccine-degradation",
    # "CDS-BART-Ecoli-proteins",
    "CDS-BART-Tc-Riboswitches",
    "CDS-BART-mRNA-stability",
]

for name in names:
    print(name)
    dset = load_dataset(f"mogam-ai/{name}", token=auth_token)
    # print(dset)


print("All datasets loaded successfully.")


pretrained_name = "mogam-ai/CDS-BART-denoising"

model = BartModel.from_pretrained(pretrained_name, token=auth_token)
assert model.config.pad_token_id == 1, (
    f"pad_token_id should be 1, but got {model.config.pad_token_id}"
)
assert model.config.bos_token_id == 0, (
    f"bos_token_id should be 0, but got {model.config.bos_token_id}"
)


print(model)

# tokenizer = BartTokenizerFast.from_pretrained(pretrained_name, token=auth_token)


# assert tokenizer.bos_token_id == 0, (
#     f"bos_token_id should be 0, but got {tokenizer.bos_token_id}"
# )
# assert tokenizer.eos_token_id == 2, (
#     f"eos_token_id should be 2, but got {tokenizer.eos_token_id}"
# )
# assert tokenizer.unk_token_id == 3, (
#     f"unk_token_id should be 3, but got {tokenizer.unk_token_id}"
# )
# assert tokenizer.sep_token_id == 2, (
#     f"sep_token_id should be 2, but got {tokenizer.sep_token_id}"
# )
# assert tokenizer.pad_token_id == 1, (
#     f"pad_token_id should be 1, but got {tokenizer.pad_token_id}"
# )
# assert tokenizer.mask_token_id == 4, (
#     f"mask_token_id should be 4, but got {tokenizer.mask_token_id}"
# )
# assert tokenizer.cls_token_id == 0, (
#     f"cls_token_id should be 0, but got {tokenizer.cls_token_id}"
# )
# assert model.config.eos_token_id == 2, (
#     f"eos_token_id should be 2, but got {model.config.eos_token_id}"
# )


# print(tokenizer)

print("Model and tokenizer loaded successfully with correct token IDs.")
