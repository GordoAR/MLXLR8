"""Propose prior prompt tokens; the target must verify every proposal."""


class PromptLookup:
    def __init__(self, tokens):
        self.tokens = []
        self.positions = {}
        self.cursor = None
        for token in tokens:
            self.append(token)

    def append(self, token):
        token = int(token)
        if self.cursor is not None:
            if self.cursor < len(self.tokens) and self.tokens[self.cursor] == token:
                self.cursor += 1
            else:
                self.cursor = None
        self.tokens.append(token)
        end = len(self.tokens) - 1
        for length in range(5, min(16, end) + 1):
            self.positions[tuple(self.tokens[end - length:end])] = end

    def propose(self, limit):
        if self.cursor is not None and self.cursor < len(self.tokens):
            return self.tokens[self.cursor:min(self.cursor + limit, len(self.tokens))]
        for length in range(min(16, len(self.tokens)), 4, -1):
            start = self.positions.get(tuple(self.tokens[-length:]))
            if start is not None:
                self.cursor = start
                return self.tokens[start:min(start + limit, len(self.tokens))]
        return []
