from django.shortcuts import render
# from django.http import HttpResponse
# Create your views here.
import torch
# model = torch.load('./savedModels/model.pt')
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
# import transformers
from transformers import GPT2TokenizerFast
import threading
import torch
import pickle
import torch.nn as nn
from deployment.softembedding import SoftEmbedding
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")

# model = T5ForConditionalGeneration.from_pretrained("t5-small")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# pickle_file='prefix-8-512-imp_sentences-1-model.pkl'

# sed = SoftEmbedding()


def tryityourself(request):
    return render(request,'tryityourself.html')

def about(request):
    return render(request,'about.html')

def architecture(request):
    return render(request,'architecture.html')

def result(request):
    return render(request,'result.html')

def askmeanything(request):
    return render(request,'askmeanything.html')

class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        """appends learned embedding to 
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                  n_tokens, 
                                                                                  random_range, 
                                                                                  initialize_from_vocab))
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
            
    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)



print('here')
with open('./savedModels/100_1_prefix_with_top_3_imp_sentences_t5_s.pkl', 'rb') as file:
    # Load the pickled object from the file
    from deployment.softembedding import SoftEmbedding
    model = pickle.load(file)


# tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def top_k_logits(logits, k,topk=0.7):
    #print('start top-k-logits')
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    #print('top-k-logits')
    return out

# def generate(examples, temperature, output_length = 120):

#     model.eval();
#     with torch.no_grad():
#         tokens = []
#     input_ids = examples['input_ids'].cuda()
#     #attention_mask = torch.tensor(examples['attention_mask']).cuda()
#     summary_ids = input_ids.clone()

#     #summary_attention = attention_mask.clone()
#     for k in range(output_length):
#         logits = model(input_ids=input_ids).logits
#         logits = logits[:, -1, :] / temperature
#         logits = top_k_logits(logits, 10)
#         probs = F.softmax(logits, dim=-1)   
#         next_token = torch.multinomial(probs, num_samples=1)
#         while next_token == tokenizer.pad_token_id:
#             next_token = torch.multinomial(probs, num_samples=1)
#         summary_ids = torch.cat([summary_ids, next_token], dim=1)
#         input_ids = summary_ids
#     summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
#     return summary


def generate(examples,decoder_input,temperature, output_length = 120):
  model.eval();
  with torch.no_grad():
    tokens = []
    input_ids = examples['input_ids'].cuda()
    #attention_mask = torch.tensor(examples['attention_mask']).cuda()
    summary_ids = input_ids.clone()

    #summary_attention = attention_mask.clone()
    for k in range(output_length):
      logits = model(input_ids=input_ids,decoder_input_ids= decoder_input.cuda()).logits
    
      logits = logits[:, -1, :] / temperature
      logits = top_k_logits(logits, 10)
      probs = F.softmax(logits, dim=-1)   
      next_token = torch.multinomial(probs, num_samples=1)
      while next_token == tokenizer.pad_token_id:
          next_token = torch.multinomial(probs, num_samples=1)
      tokens.append(next_token)
      summary_ids = torch.cat([summary_ids, next_token], dim=1)
      input_ids = summary_ids
      #summary_attention = 
      #attention_mask = torch.cat([attention_mask,torch.full((1,1), 1).cuda()],1).cuda()
    #print(summary_ids)
    summary = tokenizer.decode(torch.tensor(tokens), skip_special_tokens=True)
    #document = examples['document']
    #rouge_score = get_rouge_score(summary, document)
    return summary


def summary(request):
    received_text = request.GET['message']
    txt = '"Prison Link Cymru had 1,099 referrals in 2015-16 and said some ex-offenders were living rough for up to a year before finding suitable accommodation.\nWorkers at the charity claim investment in housing would be cheaper than jailing homeless repeat offenders.\nThe Welsh Government said more people than ever were getting help to address housing problems.\nChanges to the Housing Act in Wales, introduced in 2015, removed the right for prison leavers to be given priority for accommodation."'
    decoder_input = tokenizer(txt, return_tensors="pt").input_ids
    tokenized_text = tokenizer(received_text.strip(),max_length=512,return_tensors="pt", truncation=True)
    # print('tokenized text')
    sum = generate(tokenized_text,decoder_input,1,120)
    # print(received_text) 
    print('summmm:', sum)
    return render(request,'tryityourself.html',{'summary':sum,'source_document':received_text})




# def formInfo(request):

#     return render(request,'tryityourself.html')
