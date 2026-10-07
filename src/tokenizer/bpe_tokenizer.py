from collections import Counter
from src.tokenizer.base_tokenizer import BaseTokenizer
import pickle

class BPETokenizer(BaseTokenizer):

    def __init__(self, num_merges: int, merge_rules: list = None, token_to_id: dict = None, id_to_token: dict = None):
        super().__init__(token_to_id, id_to_token)
        self.num_merges = num_merges
        self.merge_rules = merge_rules or []
        self.token_to_id = token_to_id or {}
        self.id_to_token = id_to_token or {}

    @property
    def name(self) -> str:
        return "bpe_tokenizer"
        
    def get_pair_freqs(self, ids: list) -> Counter:
        return Counter(zip(ids, ids[1:]))
    
    def merge_pair(self, pair: tuple, tokens: list) -> list:
        a, b = pair
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
    
    def train_bpe(self, data: str):
        tokens = list(data)
        merge_rules = []
        for _ in range(self.num_merges):
            pairs = self.get_pair_freqs(tokens)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            tokens = self.merge_pair(best_pair, tokens)
            merge_rules.append(best_pair)
        self.merge_rules = merge_rules
        # rebuild vocab deterministically
        vocab = set(list(data))
        tokens = list(data)
        for pair in self.merge_rules:
            tokens = self.merge_pair(pair, tokens)
            vocab.update(tokens)
        vocab = sorted(vocab)
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def train(self, data: str) -> None:
        self.train_bpe(data)

    def encode(self, text: str) -> list[int]:
        tokens = list(text)
        for pair in self.merge_rules:
            tokens = self.merge_pair(pair, tokens)
        return [self.token_to_id[t] for t in tokens]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.id_to_token[i] for i in ids)
    
    def save_path(self) -> str:
        return f'{self.name}_{self.num_merges}'

    def save(self, folder_path: str) -> None:
        tokenizer_parameters = {'merge_rules': self.merge_rules, 
                                'token_to_id': self.token_to_id, 
                                'id_to_token': self.id_to_token}
        with open(folder_path + '/' + self.save_path() + '.pkl', "wb") as f:
            pickle.dump(tokenizer_parameters, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            tokenizer_parameters = pickle.load(f)
        self.merge_rules = tokenizer_parameters['merge_rules']
        self.token_to_id = tokenizer_parameters['token_to_id']
        self.id_to_token = tokenizer_parameters['id_to_token']
