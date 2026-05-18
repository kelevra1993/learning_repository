from collections import deque


class RecentCounter:

    def __init__(self):
        self.counter = None
        self.request_queue = deque()

    def ping(self, t: int) -> int:

        self.request_queue.append(t)
        while self.request_queue[0] < t - 3000:
            self.request_queue.popleft()

        print(self.request_queue)
        return len(self.request_queue)


exercice = RecentCounter()

for element in range(1, 10000, 500):
    exercice.ping(t=element)
