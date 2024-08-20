import os
import fitz  # PyMuPDF
import docx  # python-docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QTextEdit
from PyQt5.QtGui import QIcon, QColor

class FileSimilarityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('File Similarity Tester')
        self.setGeometry(100, 100, 800, 600)

        icon_path = os.path.join(os.path.dirname(__file__), 'icon', 'file.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print("Icon file not found")

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)

        self.load_button = QPushButton('Load Files', self)
        self.load_button.clicked.connect(self.loadFiles)

        self.reset_button = QPushButton('Reset', self)
        self.reset_button.clicked.connect(self.resetUI)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.load_button)
        self.layout.addWidget(self.reset_button)
        self.layout.addWidget(self.text_edit)
        self.setLayout(self.layout)

    def loadFiles(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        options |= QFileDialog.ExistingFiles

        files, _ = QFileDialog.getOpenFileNames(self, 'Select Files', '', 
                                            "All Files (*)", 
                                            options=options)

        print("Selected files:", files)  

        if files:
            self.processFiles(files)




    def resetUI(self):
        self.text_edit.clear()

    def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def extract_text_from_txt(self, txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return text
        except Exception as e:
            print(f"Error reading {txt_path}: {e}")
            return ""

    def extract_text_from_docx(self, docx_path):
        try:
            doc = docx.Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error reading {docx_path}: {e}")
            return ""

    def processFiles(self, files):
        texts = []
        for file in files:
            if file.lower().endswith('.pdf'):
                texts.append(self.extract_text_from_pdf(file))
            elif file.lower().endswith('.txt'):
                texts.append(self.extract_text_from_txt(file))
            elif file.lower().endswith('.docx'):
                texts.append(self.extract_text_from_docx(file))
            else:
                print(f"Unsupported file type: {file}")

        if not texts:
            print("No text extracted from files.")
            return

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

        cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

        similarities = []
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                similarity = cosine_similarities[i][j] * 100
                similarity = round(similarity, 4)
                pair = (os.path.basename(files[i]), os.path.basename(files[j]), similarity)
                similarities.append(pair)

        similarities.sort(key=lambda x: x[2], reverse=True)

        custom_dim_gray = QColor(20, 20, 20)
        custom_red = QColor(255, 0, 0)

        result_text = "<span style='color: red;'>File Similarity Results:</span><br>"
        for pair in similarities:
            similarity_percent = f"{pair[2]}%"
            result_text += f"{pair[0]} - {pair[1]} : Similarity " \
                           f"<span style='color: {custom_red.name()}; background-color: {custom_dim_gray.name()};'>{similarity_percent}</span><br>"

        self.text_edit.setHtml(result_text)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = FileSimilarityApp()

    window.setStyleSheet("background-color: rgb(105, 105, 105); color: white;")
    window.text_edit.setStyleSheet("color: white;")

    window.show()
    sys.exit(app.exec_())
