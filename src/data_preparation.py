import numpy as np
import os

def normalize_data(x):
    norms = np.linalg.norm(x, axis=0)
    norms[norms == 0] = 1.0
    return x / norms

def main():
    data_dir = "./data"
    files = ["TrainDigits.npy", "TestDigits.npy"]
    
    for f in files:
        path = os.path.join(data_dir, f)
        x = np.load(path)
        np.save(path, normalize_data(x))

if __name__ == "__main__":
    main()
