from config import MEMORY_WINDOW

class Memory:
    def __init__(self):
        self.history = []

    def add(self, role, content):
        self.history.append((role, content))
        if len(self.history) > MEMORY_WINDOW:
            self.history = self.history[-MEMORY_WINDOW:]

    def get(self):
        return "\n".join([f"{r}: {c}" for r,c in self.history])