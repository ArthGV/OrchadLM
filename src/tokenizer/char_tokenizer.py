from src.tokenizer.base_tokenizer import BaseTokenizer
import pickle

class CharTokenizer(BaseTokenizer):

    def __init__(self, token_to_id: dict = None, id_to_token: dict = None):
        super().__init__(token_to_id, id_to_token)
    
    @property
    def name(self) -> str:
        return "char_tokenizer"

    def train(self, data: str) -> None:
        alphabet = sorted(set(data)) 
        self.token_to_id = {a: i for i, a in enumerate(alphabet)}
        self.id_to_token = {i: a for i, a in enumerate(alphabet)}

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id[t] for t in text]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.id_to_token[i] for i in ids)
    
    def save_path(self) -> str:
        return f'{self.name}'

    def save(self, folder_path: str) -> None:
        tokenizer_parameters = {'token_to_id': self.token_to_id, 
                                'id_to_token': self.id_to_token}
        with open(folder_path + '/' + self.save_path() + '.pkl', "wb") as f:
            pickle.dump(tokenizer_parameters, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            tokenizer_parameters = pickle.load(f)
        self.token_to_id = tokenizer_parameters['token_to_id']
        self.id_to_token = tokenizer_parameters['id_to_token']
