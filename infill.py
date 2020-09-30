# Imports
import os.path
import pickle
import ilm.tokenize_util
from transformers import GPT2LMHeadModel
from ilm.infer import infill_with_ilm
import gdown

# Variables
MODEL_DIR = 'model/'
MASK_CLS = 'ilm.mask.hierarchical.MaskHierarchical'
result = []
tokenizer = ilm.tokenize_util.Tokenizer.GPT2

datamodel = 'model/pytorch_model.bin'
model_location = "https://drive.google.com/uc?id=1-12EFaKNBYD1vlfeZcKnV5PaSqeHNTHX"

if os.path.isfile(datamodel):
    ('Model was already downloaded.')
else:
    gdown.download(model_location, datamodel)

# Create context
context = 'The sun is shining. _ All the children want to swim.'


class INFILL:
    def infilling(self, context: str):
        with open(os.path.join(MODEL_DIR, 'additional_ids_to_tokens.pkl'), 'rb') as f:
            additional_ids_to_tokens = pickle.load(f)
        additional_tokens_to_ids = {v: k for k, v in additional_ids_to_tokens.items()}
        try:
            ilm.tokenize_util.update_tokenizer(additional_ids_to_tokens, tokenizer)
        except ValueError:
            print('Already updated')

        # Load model
        device = 'cpu'
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
        model.eval()
        _ = model.to(device)
        context_ids = ilm.tokenize_util.encode(context, tokenizer)
        _blank_id = ilm.tokenize_util.encode(' _', tokenizer)[0]
        context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_sentence|>']

        generated = infill_with_ilm(
            model,
            additional_tokens_to_ids,
            context_ids,
            num_infills=5)
        for g in generated:
            # print('-' * 80)
            # print(ilm.tokenize_util.decode(g, tokenizer))
            result.append(str(ilm.tokenize_util.decode(g, tokenizer)))
        print(result)
        return result
