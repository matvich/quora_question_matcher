# %%
import string
from collections import Counter
from typing import Dict, List, Tuple, Union, Callable
 
import os
import nltk
import numpy as np
import math
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.modules.sparse import Embedding

from tqdm.auto import tqdm

from flask import Flask, request, jsonify
import faiss
import langdetect
import json

class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        # допишите ваш код здесь 
        return torch.exp(-(x - self.mu)**2/(2*self.sigma**2))


EMB_PATH_GLOVE = os.environ['EMB_PATH_GLOVE']
EMB_PATH_KNRM = os.environ['EMB_PATH_KNRM']
VOCAB_PATH = os.environ['VOCAB_PATH']
MLP_PATH = os.environ['MLP_PATH']

# EMB_PATH_GLOVE='../B2_HW5_L5/glove.6B/glove.6B.50d.txt' 
# EMB_PATH_KNRM='./knrm_emb.bin'
# VOCAB_PATH='./vocab.json'
# MLP_PATH='./user_mlp.bin'

class KNRM(torch.nn.Module):
    def __init__(self, knrm_embeddings, mlp_state_dict, kernel_num = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(knrm_embeddings),
            freeze=True,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers
        self.kernels = self._get_kernels_layers()
        self.mlp = self._get_mlp()
        self.mlp.load_state_dict(mlp_state_dict)
        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        mus = np.convolve(np.linspace(-1, 1, self.kernel_num), np.array([0.5, 0.5]))[1:-1]
        for mu in mus:
            kernels.append(GaussianKernel(mu, self.sigma))
        kernels.append(GaussianKernel(1, self.exact_sigma))
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
       layers = []
       prev_hid_dim = self.kernel_num
       for hid_dim in self.out_layers:
           layers.append(torch.nn.ReLU())
           layers.append(torch.nn.Linear(prev_hid_dim, hid_dim))
           prev_hid_dim = hid_dim
       layers.append(torch.nn.Linear(prev_hid_dim, 1))
       return torch.nn.Sequential(*layers)

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        # допишите ваш код здесь 
        query_emb = self.embeddings(query)
        doc_emb = self.embeddings(doc)
        q_norm = F.normalize(query_emb, p=2, dim=-1)
        d_norm = F.normalize(doc_emb, p=2, dim=-1)
        dot_prod = torch.bmm(q_norm, d_norm.transpose(1, 2))
        return dot_prod #/ torch.max(1e-14 * torch.ones_like(dot_prod), q_norm.unsqueeze(2) * d_norm.unsqueeze(1))

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        query, doc = inputs['query'], inputs['document']
        matching_matrix = self._get_matching_matrix(query, doc)
        kernels_out = self._apply_kernels(matching_matrix)
        out = self.mlp(kernels_out)
        return out


class Solution:
    def __init__(self, 
                 random_seed: int = 0,
                ):

        self.random_seed = random_seed
        self.model_ready = False
        self.index_ready = False
        self.model, self.vocab = self._build_knrm_model()

    @property     
    def is_model_ready(self):
        return self.model_ready

    @property
    def is_index_ready(self):
        return self.index_ready
 
    def handle_punctuation(self, inp_str: str) -> str:
        # допишите ваш код здесь 
        for char in string.punctuation:
            inp_str = inp_str.replace(char, ' ')
        return inp_str

    def simple_preproc(self, inp_str: str) -> List[str]:
        # допишите ваш код здесь 
        str_no_punct = self.handle_punctuation(inp_str)
        lower_str = str_no_punct.lower()
        tokens = nltk.word_tokenize(lower_str)
        return tokens
    
    def _create_index(self, documents: Dict[str, str], vocabulary):
        self.documents = documents
        embeddings = self.model.embeddings.state_dict()['weight'].numpy()
        idx_list, doc_emb_list = [], []
        for idx, document in documents.items():
            tokens = self.simple_preproc(document)
            ids = list(map(lambda x: vocabulary.get(x, vocabulary['OOV']), tokens))
            token_embs = embeddings[ids, :]
            doc_emb = token_embs.mean(axis=0)
            idx_list.append(int(idx))
            doc_emb_list.append(doc_emb)

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index = faiss.IndexIDMap(self.index)
        self.index.add_with_ids(np.array(doc_emb_list), np.array(idx_list).astype(int))
        self.index_ready = True
        return self.index.ntotal

    def _build_knrm_model(self) -> Tuple[torch.nn.Module, Dict[str, int]]:
        torch.manual_seed(self.random_seed)
        with open(VOCAB_PATH) as file:
            vocab = json.load(file)
        knrm = KNRM(torch.load(EMB_PATH_KNRM)['weight'],
                    torch.load(MLP_PATH),
                    out_layers=[])
        self.model_ready = True
        return knrm, vocab

    def _text2tokens(self, texts):
        max_len = max(map(len, texts))
        token_ids = []
        for text in texts:
            tokens = self.simple_preproc(text)
            ids = list(map(lambda x: self.vocab.get(x, self.vocab['OOV']), tokens))
            ids = ids[:max_len] + [0]*(max_len - len(ids))
            token_ids.append(ids)
        return torch.LongTensor(token_ids)

    def get_suggestions(self, query: str):
        embeddings = self.model.embeddings.state_dict()['weight'].numpy()
        tokens = self.simple_preproc(query)
        ids = list(map(lambda x: self.vocab.get(x, self.vocab['OOV']), tokens))
        token_embs = embeddings[ids, :]
        q_emb = token_embs.mean(axis=0)#.astype(np.float32)
        # print(q_emb)
        _, inds = self.index.search(q_emb[None], 30)
        # print(inds)
        candidates = [(str(i), self.documents[str(i)]) for i in inds[0] if i != -1]
        inputs = dict()
        inputs['query'] = self._text2tokens([query] * len(candidates))
        inputs['document'] = self._text2tokens([cnd[1] for cnd in candidates])
        # print(inputs)
        scores = self.model(inputs).squeeze()
        sort_ids = scores.argsort(descending=True)[:10].tolist()
        return [candidates[i] for i in sort_ids]

app = Flask(__name__)

sol = Solution()

@app.route('/ping')
def ping():
    if sol.is_model_ready:
        return jsonify(status='ok')

@app.route('/update_index', methods=['POST'])
def update_index():
    # print(request.json)
    req = json.loads(request.json)
    # print(req)
    # req = request.json
    documents = req['documents']
    return jsonify(status='ok', index_size=sol._create_index(documents, sol.vocab))

@app.route('/query', methods=['POST'])
def query():
    if not sol.index_ready:
        return jsonify(status='FAISS is not initialized!')
    else:
        req = json.loads(request.json)
        # req = request.json
        queries = req['queries']
        lang_check = []
        suggestions = []
        for query in queries:
            is_eng = langdetect.detect(query) == 'en'
            lang_check.append(is_eng)
            if not is_eng:
                suggestions.append(None)
            else:
                suggestions.append(sol.get_suggestions(query))
        return jsonify(lang_check=lang_check,
                       suggestions=suggestions)


# %%
