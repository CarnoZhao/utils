import time
import numpy as np
import pandas as pd
import transformers

# tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

class Tokenize():
    def __init__(self, name, *args, **kwargs):
        self.tokenizer = transformers.BertTokenizer.from_pretrained(name, *args, **kwargs)

    def __call__(self, x, y = None, max_len = 128):
        x = x.replace(" ", "")
        x = self.tokenizer.encode_plus(
            x,
            y,
            add_special_tokens=True,
            truncation = 'longest_first',
            max_length = max_len,
            padding="max_length"
        )
        ret = {}
        ret["ids"] = np.array(x["input_ids"]).astype(np.long)
        ret["mask"] = np.array(x["attention_mask"]).astype(np.long)
        ret["token_type_ids"] = np.array(x["token_type_ids"]).astype(np.long)
        return ret