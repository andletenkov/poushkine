from pathlib import Path


class Text2Vec:
    """Simple character-level text tokenizer."""

    def __init__(self) -> None:
        self.char_to_index = {}
        self.index_to_char = {}

    def fit(self, file_path: Path | str) -> None:
        """Fits the tokenizer.

        :param file_path: Path to a file with text.
        """
        text = Path(file_path).read_text()
        chars = "".join(sorted(set(text)))

        for idx, char in enumerate(chars):
            self.char_to_index[char] = idx
            self.index_to_char[idx] = char

    def encode(self, text: str) -> list[int]:
        """Encodes text to list of integer tokens.

        :param text: Any string.
        :return: Tokens list.
        """
        try:
            return [self.char_to_index[char] for char in text]
        except KeyError as e:
            raise ValueError(f"Unknown character {e}. Forgot to fit?") from None

    def decode(self, tokens: list[int]) -> str:
        """Decodes tokens to a original string.

        :param tokens: List of integer tokens.
        :return: Text.
        """
        try:
            return "".join([self.index_to_char[token] for token in tokens])
        except KeyError as e:
            raise ValueError(f"Unknown token {e}. Forgot to fit?") from None
