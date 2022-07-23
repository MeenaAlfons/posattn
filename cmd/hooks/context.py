class Context:
    def __init__(self):
        self._context = {}

    def set(self, key, value):
        self._context[key] = value

    def get(self, key):
        return self._context[key]

    def __setitem__(self, key, val):
        self._context[key] = val

    def __getitem__(self, key):
        return self._context[key]
