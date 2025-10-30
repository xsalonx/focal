import pickle
import os

def save_centroids(path, centroids):
    os.makedirs(path, exist_ok=True)
    with open(path / 'centroids-data.pkl', 'wb') as f:
        pickle.dump(centroids, f)

def load_centroids(path):
    with open(path / 'centroids-data.pkl', 'rb') as f:
        return pickle.load(f)