import os
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
print("\033[92m PDF File Similarity Tester \033[0m")

# PDF 파일을 텍스트로 추출
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# PDF 파일 목록 가져오기
pdf_folder = 'files'
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

# PDF 파일들의 텍스트 추출
texts = [extract_text_from_pdf(os.path.join(pdf_folder, pdf_file)) for pdf_file in pdf_files]

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

# 유사도 계산 (코사인 유사도)
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 쌍별 유사도 저장
similarities = []
for i in range(len(pdf_files)):
    for j in range(i + 1, len(pdf_files)):
        similarity = cosine_similarities[i][j] * 100  # 백분율로 표시
        similarity = round(similarity, 4)  # 소수점 이하 두 자리까지 표시
        stringsimilarity = "\033[91m" + str(similarity) + "\033[0m"
        pair = (pdf_files[i], pdf_files[j], stringsimilarity)
        similarities.append(pair)

# 유사도가 높은 순으로 정렬
similarities.sort(key=lambda x: x[2], reverse=True)

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

print("\033[92m 이름1         이름2         유사도\033[0m")
for pair in similarities:
    print(f"{pair[0]}     {pair[1]}     {pair[2]}")
