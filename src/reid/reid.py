import cv2
import numpy as np

class ReIdentifier:
    def __init__(self):
        self.database = []

    def extract_feature(self, frame, l, t, w, h):
        crop = frame[int(t):int(t+h), int(l):int(l+w)]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, (32, 32))
        return crop.flatten()

    def add(self, feature):
        self.database.append(feature)

    def match(self, feature, threshold=2000):
        for f in self.database:
            if np.linalg.norm(feature - f) < threshold:
                return True
        return False