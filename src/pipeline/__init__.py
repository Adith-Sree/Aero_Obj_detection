
from src.tracking.tracker import Tracker

class UAVPipeline:
    def __init__(self):
        self.detector = Detector("models/weights/best.pt")
        self.tracker = Tracker()   # 🔥 THIS LINE WAS MISSING
        self.reid = ReIdentifier()
        self.behavior = BehaviorAnalyzer()

        self.flagged_ids = set()