import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.sum(np.abs(a - b))

def vector_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b

def vector_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a - b

def vector_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2

def vector_dot(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b)

def vector_norm(a: np.ndarray) -> float:
    return np.linalg.norm(a)