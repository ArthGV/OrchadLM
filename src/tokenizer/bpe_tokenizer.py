from collections import Counter
from src.tokenizer.base_tokenizer import BaseTokenizer
import pickle

class BPETokenizer(BaseTokenizer):

    def __init__(self, num_merges: int, merge_rules: list = None, token_to_id: dict = None, id_to_token: dict = None):
        super().__init__(token_to_id, id_to_token)
        self.num_merges = num_merges
        self.merge_rules = merge_rules or []
        
    def get_pair_freqs(self, ids: list) -> Counter:
        return Counter(zip(ids, ids[1:]))
    
    def merge_pair(self, pair: tuple, ids: list) -> list:
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(''.join(pair))
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def train_bpe(self, data: str, num_merges: int):
        ids = list(data)
        merge_rules = []

        for _ in range(num_merges):
            pairs = self.get_pair_freqs(ids)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            ids = self.merge_pair(best, ids)
            merge_rules.append(best)

        # Build mappings
        all_tokens  = sorted(set(ids) | set(data))   # include base chars for encoding unseen text
        token_to_id = {t: i for i, t in enumerate(all_tokens)}
        id_to_token = {i: t for t, i in token_to_id.items()}

        self.merge_rules = merge_rules
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token

    def train(self, data: str) -> None:
        self.train_bpe(data, self.num_merges)

    def encode(self, text: str) -> list[int]:
        ids = list(text)
        for pair in self.merge_rules:
            ids = self.merge_pair(pair, ids)
        return [self.token_to_id[t] for t in ids]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.id_to_token[i] for i in ids)

    def save(self, path: str) -> None:
        tokenizer_parameters = {'merge_rules': self.merge_rules, 
                                'token_to_id': self.token_to_id, 
                                'id_to_token': self.id_to_token}
        with open(path, "wb") as f:
            pickle.dump(tokenizer_parameters, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            tokenizer_parameters = pickle.load(f)
        self.merge_rules = tokenizer_parameters['merge_rules']
        self.token_to_id = tokenizer_parameters['token_to_id']
        self.id_to_token = tokenizer_parameters['id_to_token']
