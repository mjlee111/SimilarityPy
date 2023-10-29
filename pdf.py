import os
import fitz  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
print("\033[92m PDF File Similarity Tester \033[0m")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

pdf_folder = 'files'
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

texts = [extract_text_from_pdf(os.path.join(pdf_folder, pdf_file)) for pdf_file in pdf_files]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

similarities = []
for i in range(len(pdf_files)):
    for j in range(i + 1, len(pdf_files)):
        similarity = cosine_similarities[i][j] * 100 
        similarity = round(similarity, 4)  
        stringsimilarity = "\033[91m" + str(similarity) + "\033[0m"
        pair = (pdf_files[i], pdf_files[j], stringsimilarity)
        similarities.append(pair)

similarities.sort(key=lambda x: x[2], reverse=True)

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

print("\033[92m 이름1         이름2         유사도\033[0m")
for pair in similarities:
    print(f"{pair[0]}     {pair[1]}    {pair[2]}")
