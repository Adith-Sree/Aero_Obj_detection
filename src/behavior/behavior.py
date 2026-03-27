import numpy as np

class BehaviorAnalyzer:
    def __init__(self):
        self.track_history = {}

    def update(self, track_id, center):
        if track_id not in self.track_history:
            self.track_history[track_id] = []

        self.track_history[track_id].append(center)

        if len(self.track_history[track_id]) > 30:
            self.track_history[track_id].pop(0)

    def is_stationary(self, track_id, threshold=10):
        points = self.track_history.get(track_id, [])

        if len(points) < 10:
            return False

        distances = [
            np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))
            for i in range(1, len(points))
        ]

        return np.mean(distances) < threshold