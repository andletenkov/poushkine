class Text2Vec:
    """Simple character-level text tokenizer."""

    def __init__(self) -> None:
        self.char_to_index = {}
        self.index_to_char = {}
        self._vocab_size = 0

    @property
    def vocab_size(self) -> int:
        """Return number of characters embedded in tokenizer.

        :return: Integer.
        """
        return self._vocab_size

    def fit(self, text: str) -> None:
        """Fits the tokenizer.

        :param text: Text data to fit the tokenizer.
        """
        chars = "".join(sorted(set(text)))
        self._vocab_size = len(chars)

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
