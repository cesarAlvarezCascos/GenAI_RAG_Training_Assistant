from collections import defaultdict, deque

class SessionMemory:
    def __init__(self, max_turns: int = 3):
        # Each user gets a deque of (query, answer) pairs
        self.history = defaultdict(lambda: deque(maxlen=max_turns))

    # Older interactions are dropped/deleted if max_turns limit is surpassed
    def add(self, user_id: str, query: str, answer: str):
        self.history[user_id].append({"query": query, "answer": answer})

    def get(self, user_id: str):
        return list(self.history[user_id])

    def clear(self, user_id: str):
        self.history.pop(user_id, None)
