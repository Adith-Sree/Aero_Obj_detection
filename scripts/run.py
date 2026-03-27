import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.pipeline import UAVPipeline

if __name__ == "__main__":
    app = UAVPipeline()
    app.run()