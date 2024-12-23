from flask import Flask, render_template, request
import os
import re
import math
from collections import Counter
from werkzeug.utils import secure_filename
import tempfile
import docx
import PyPDF2
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

ALLOWED_EXTENSIONS = {'txt', 'doc', 'docx', 'pdf'}

# Fungsi untuk membaca kamus stemming
def load_stemming_dict():
    stemming_dict = {}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dict_path = os.path.join(current_dir, 'kamuss.txt')
    
    with open(dict_path, 'r', encoding='utf-8') as file:
        for line in file:
            if '|' in line:
                num, word = line.strip().split('|', 1)
                word = word.strip()
                if word:
                    stemming_dict[word] = word
    return stemming_dict

# Load kamus stemming
stemming_dict = load_stemming_dict()

def stem_word(word):
    """
    Fungsi untuk mencari kata dasar menggunakan kamus
    """
    word = word.lower()
    if word in stemming_dict:
        return stemming_dict[word]
    return word

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_txt_file(file):
    return file.read().decode('utf-8')

def read_docx_file(file):
    doc = docx.Document(file)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return ' '.join(text)

def read_pdf_file(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = []
    for page in pdf_reader.pages:
        text.append(page.extract_text())
    return ' '.join(text)

def read_file_content(file):
    filename = file.filename.lower()
    if filename.endswith('.txt'):
        return read_txt_file(file)
    elif filename.endswith('.docx'):
        return read_docx_file(file)
    elif filename.endswith('.pdf'):
        return read_pdf_file(file)
    return ""

def preprocess_text(text):
    # Case folding
    text = text.lower()
    
    # Tokenizing dan filtering (hanya ambil kata-kata)
    tokens = re.findall(r'\b\w+\b', text)
    
    # Stopword removal (tambahkan stopwords sesuai kebutuhan)
    stopwords = set(['yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 'dan', 'atau'])
    tokens = [token for token in tokens if token not in stopwords]
    
    # Stemming menggunakan kamus
    stemmed_tokens = [stem_word(token) for token in tokens]
    
    return stemmed_tokens

def calculate_tf_idf(documents, query):
    # Menghitung TF untuk setiap dokumen
    doc_tf = [Counter(doc) for doc in documents]
    query_tf = Counter(query)
    
    # Menghitung IDF
    N = len(documents)
    word_set = set().union(*documents)
    idf = {}
    
    for word in word_set:
        doc_count = sum(1 for doc in documents if word in doc)
        idf[word] = math.log10(N / doc_count) if doc_count > 0 else 0
    
    # Menghitung TF-IDF vectors
    doc_vectors = []
    for tf in doc_tf:
        vector = {word: tf[word] * idf.get(word, 0) for word in word_set}
        doc_vectors.append(vector)
    
    query_vector = {word: query_tf[word] * idf.get(word, 0) for word in word_set}
    
    return doc_vectors, query_vector

def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    return float(numerator) / denominator

@app.route('/', methods=['GET', 'POST'])
def index():
    query = None
    original_query = None
    processed_query = None
    similarities = []
    documents_data = []

    if request.method == 'POST':
        query = request.form['query']
        
        if 'documents[]' not in request.files:
            return render_template('index.html', error='Tidak ada file yang dipilih')
        
        files = request.files.getlist('documents[]')
        
        # Proses setiap file yang diupload
        documents = []
        doc_names = []
        original_tokens = []  # Untuk menyimpan token sebelum stemming
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                try:
                    content = read_file_content(file)
                    if content:
                        # Simpan token original
                        tokens = re.findall(r'\b\w+\b', content.lower())
                        original_tokens.append(tokens)
                        # Proses stemming
                        documents.append(preprocess_text(content))
                        doc_names.append(filename)
                except Exception as e:
                    print(f"Error membaca file {filename}: {str(e)}")
                    continue
        
        if not documents:
            return render_template('index.html', error='Tidak ada file valid yang diupload')
        
        # Preprocess query
        original_query = re.findall(r'\b\w+\b', query.lower())
        processed_query = preprocess_text(query)
        
        # Hitung VSM
        doc_vectors, query_vector = calculate_tf_idf(documents, processed_query)
        
        # Hitung similarity
        for i, doc_vector in enumerate(doc_vectors):
            sim = cosine_similarity(doc_vector, query_vector)
            similarities.append((doc_names[i], sim))
        
        # Sort berdasarkan similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        documents_data = list(zip(doc_names, original_tokens, documents))
    
    return render_template('index.html', 
                           query=query,
                           original_query=original_query,
                           processed_query=processed_query,
                           similarities=similarities,
                           documents=documents_data)

if __name__ == '__main__':
    app.run(debug=True) 