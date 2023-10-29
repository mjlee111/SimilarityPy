import os
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QTextEdit
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtCore import Qt

class PDFSimilarityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('PDF File Similarity Tester')
        self.setGeometry(100, 100, 800, 600)

        icon_path = os.path.join(os.path.dirname(__file__), 'icon', 'pdf.png')
        self.setWindowIcon(QIcon(icon_path))

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)

        self.load_button = QPushButton('Load PDF Files', self)
        self.load_button.clicked.connect(self.loadPDFFiles)

        self.reset_button = QPushButton('Reset', self)
        self.reset_button.clicked.connect(self.resetUI)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.load_button)
        self.layout.addWidget(self.reset_button)
        self.layout.addWidget(self.text_edit)
        self.setLayout(self.layout)

    def loadPDFFiles(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        options |= QFileDialog.ExistingFiles

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("PDF Files (*.pdf)")
        files, _ = file_dialog.getOpenFileNames(self, 'Select PDF Files', '', "PDF Files (*.pdf)", options=options)

        if files:
            self.processPDFFiles(files)

    def resetUI(self):
        self.text_edit.clear()

    def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def processPDFFiles(self, pdf_files):
        texts = [self.extract_text_from_pdf(pdf_file) for pdf_file in pdf_files]

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

        cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

        similarities = []
        for i in range(len(pdf_files)):
            for j in range(i + 1, len(pdf_files)):
                similarity = cosine_similarities[i][j] * 100
                similarity = round(similarity, 4)
                pair = (os.path.basename(pdf_files[i]), os.path.basename(pdf_files[j]), similarity)
                similarities.append(pair)

        similarities.sort(key=lambda x: x[2], reverse=True)

        custom_dim_gray = QColor(20, 20, 20)

        custom_red = QColor(255, 255, 0)

        result_text = "<span style='color: red;'>PDF Similarity Results:</span><br>"
        for pair in similarities:
            similarity_percent = f"{pair[2]}%"
            result_text += f"{pair[0]} - {pair[1]} : Similarity " \
                f"<span style='color: {custom_red.name()}; backg0nd-color: {custom_dim_gray.name()};'>{similarity_percent}</span><br>"

        self.text_edit.setHtml(result_text)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = PDFSimilarityApp()

    window.setStyleSheet("background-color: rgb(105, 105, 105); color: white;")

    window.text_edit.setStyleSheet("color: white;")

    window.show()
    sys.exit(app.exec_())
