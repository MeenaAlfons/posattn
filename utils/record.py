class _RecordClass:
    """
    Use `record` to record the values of some parameters and access them later
    """
    def __init__(self):
        self.enabled = False
        self.items = {}

    def __call__(self, name, value):
        if not self.enabled:
            return

        self.items[name] = value

    def __getitem__(self, name):
        return self.items[name]

    def clear(self):
        self.items = {}


record = _RecordClass()
"""
Use `record` to record the values of some parameters and access them later

Example::

    # If it is not enabled, the values will not be saved.
    record.enabled = True

    # This will save the value of q
    record('q', q)

    # Later you can access it with:
    record['q']
"""
