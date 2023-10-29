import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

def get_files(path):
    files = os.listdir(path)
    return [os.path.join(path, f) for f in files if os.path.isfile(os.path.join(path, f)) and f.endswith(".txt")]

def get_text(file):
    with open(file, "r") as f:
        text = f.read()
    return text

def get_embeddings(files):
    vectorizer = TfidfVectorizer()
    text = [get_text(file) for file in files]
    text_embeddings = vectorizer.fit_transform(text)
    return text_embeddings.toarray()

def get_similarity(embeddings, method="cosine"):
    if method == "cosine":
        return np.dot(embeddings, embeddings.T)
    elif method == "euclidean":
        return pairwise_distances(embeddings, metric='euclidean')
    elif method == "jaccard":
        # Calculate Jaccard similarity properly
        num_docs = embeddings.shape[0]
        jaccard_similarity = np.zeros((num_docs, num_docs))
        for i in range(num_docs):
            for j in range(i + 1, num_docs):
                intersection = np.logical_and(embeddings[i], embeddings[j]).sum()
                union = np.logical_or(embeddings[i], embeddings[j]).sum()
                jaccard_similarity[i, j] = intersection / union
                jaccard_similarity[j, i] = jaccard_similarity[i, j]
        return jaccard_similarity
    else:
        raise ValueError("지원되지 않는 유사도 측정 방법입니다.")




def main():
    RED = "\033[91m"
    RESET = "\033[0m"

    files = get_files("files")
    text_embeddings = get_embeddings(files)

    for method in ["cosine", "euclidean", "jaccard"]:
        similarities = get_similarity(text_embeddings, method)
        print(f"\033[92m{method.capitalize()} 유사도 검출\033[0m")
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                similarity = round(similarities[i][j] * 100, 2)
                print(f"{files[i]} - {files[j]}: {RED}{similarity} {RESET}%")

if __name__ == "__main__":
    main()
