from abc import ABC, abstractmethod

class BaseTokenizer(ABC):

    def __init__(self, token_to_id: dict = None, id_to_token: dict = None):
        self.token_to_id = token_to_id or {}
        self.id_to_token = id_to_token or {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name"""
        ...
    
    @abstractmethod
    def save_path(self) -> str:
        """Given hyparameters, build the save path."""
        ...

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @abstractmethod
    def train(self, data: str) -> None:
        """Train the tokenizer on raw text."""
        ...

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of token ids."""
        ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Convert a list of token ids back to a string."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Save tokenizer parameters as pickle."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load tokenizer parameters."""
        ...