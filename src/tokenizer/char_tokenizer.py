from src.tokenizer.base_tokenizer import BaseTokenizer

class CharTokenizer(BaseTokenizer):

    def __init__(self, token_to_id: dict = None, id_to_token: dict = None):
        super().__init__(token_to_id, id_to_token)        

    def train(self, data: str) -> None:
        alphabet = sorted(set(data)) 
        self.token_to_id = {a: i for i, a in enumerate(alphabet)}
        self.id_to_token = {i: a for i, a in enumerate(alphabet)}

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id[t] for t in text]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.id_to_token[i] for i in ids)
