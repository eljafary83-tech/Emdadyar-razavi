from __future__ import annotations
import numpy as np

class FakeEmbeddings:
    model_name = "fake"
    dimension = 2
    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = np.array([1.0, 0.0]) if "امداد" in text else np.array([0.0, 1.0])
            vectors.append(vector / np.linalg.norm(vector))
        return np.asarray(vectors, dtype="float32")
