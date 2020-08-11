import logging
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import torch

logging.getLogger().setLevel(logging.CRITICAL)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


class model:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        self.model = self.model.to(device)

    def display_model(self):
        print(self.model)

    def select_tokens(self, probs, n=5):
        z = np.argpartition(probs, -n)[-n:]
        top = probs[z]
        top = top / np.sum(top)
        choice = np.random.choice(n, 1, p=top)
        tkn = z[choice][0]
        return int(tkn)

    def generate_some_text(self, input_str, text_len=1):
        cur_ids = torch.tensor(self.tokenizer.encode(
            input_str)).unsqueeze(0).long().to(device)
        self.model.eval()
        with torch.no_grad():
            for i in range(text_len):
                outputs = self.model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                softmax_logits = torch.softmax(logits[0, -1], dim=0)
                next_token_id = self.select_tokens(
                    softmax_logits.to('cpu').numpy(), n=9)
                cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(
                    device) * next_token_id], dim=1)
            output_list = list(cur_ids.squeeze().to('cpu').numpy())
            output_text = self.tokenizer.decode(output_list)
            print(output_text)
            return output_text
