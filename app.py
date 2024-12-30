from flask import Flask, render_template, request  # Import Flask untuk membuat aplikasi web
import os  # Untuk operasi file dan direktori
import re  # Untuk pencocokan pola (regex)
import math  # Untuk perhitungan matematika (logaritma, akar)
from collections import Counter  # Untuk menghitung frekuensi kata
from werkzeug.utils import secure_filename  # Untuk mengamankan nama file upload
import tempfile  # Untuk membuat direktori sementara
import docx  # Untuk membaca file .docx
import PyPDF2  # Untuk membaca file PDF
import io  # Untuk manipulasi file berbasis buffer

# Inisialisasi aplikasi Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maksimal ukuran file yang diupload: 16 MB
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()  # Folder sementara untuk menyimpan file yang diupload

# Ekstensi file yang diperbolehkan
ALLOWED_EXTENSIONS = {'txt', 'doc', 'docx', 'pdf'}

# Fungsi untuk membaca kamus stemming
def load_stemming_dict():
    """
    Memuat file kamus stemming dari direktori proyek.
    File kamus digunakan untuk menemukan kata dasar dari kata tertentu.
    """
    stemming_dict = {}
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Mendapatkan direktori file saat ini
    dict_path = os.path.join(current_dir, 'kamuss.txt')  # Lokasi file kamus
    
    with open(dict_path, 'r', encoding='utf-8') as file:
        for line in file:  # Membaca file per baris
            if '|' in line:  # Pastikan formatnya sesuai (menggunakan '|')
                num, word = line.strip().split('|', 1)  # Pisahkan nomor dan kata
                word = word.strip()  # Menghapus spasi berlebih
                if word:
                    stemming_dict[word] = word  # Tambahkan ke dictionary
    return stemming_dict

# Memuat kamus stemming ke dalam variabel global
stemming_dict = load_stemming_dict()

# Fungsi untuk melakukan stemming pada kata
def stem_word(word):
    """
    Mengembalikan kata dasar menggunakan kamus stemming.
    Jika tidak ditemukan, kembalikan kata asli.
    """
    word = word.lower()  # Ubah kata menjadi huruf kecil
    if word in stemming_dict:  # Periksa apakah kata ada di kamus
        return stemming_dict[word]  # Kembalikan kata dasar
    return word  # Jika tidak ditemukan, kembalikan kata asli

# Fungsi untuk memeriksa ekstensi file yang diizinkan
def allowed_file(filename):
    """
    Mengecek apakah file memiliki ekstensi yang diperbolehkan.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi untuk membaca file teks
def read_txt_file(file):
    """
    Membaca konten file teks (.txt) dan mengembalikan sebagai string.
    """
    return file.read().decode('utf-8')

# Fungsi untuk membaca file Word (.docx)
def read_docx_file(file):
    """
    Membaca konten file .docx dan mengembalikan sebagai string.
    """
    doc = docx.Document(file)  # Membuka file dokumen
    text = []
    for paragraph in doc.paragraphs:  # Iterasi melalui setiap paragraf
        text.append(paragraph.text)  # Tambahkan teks ke daftar
    return ' '.join(text)  # Gabungkan teks menjadi satu string

# Fungsi untuk membaca file PDF
def read_pdf_file(file):
    """
    Membaca konten file PDF dan mengembalikan sebagai string.
    """
    pdf_reader = PyPDF2.PdfReader(file)  # Membuka file PDF
    text = []
    for page in pdf_reader.pages:  # Iterasi melalui setiap halaman
        text.append(page.extract_text())  # Ekstrak teks dari halaman
    return ' '.join(text)  # Gabungkan teks menjadi satu string

# Fungsi untuk membaca konten file berdasarkan jenisnya
def read_file_content(file):
    """
    Membaca konten file berdasarkan tipe file (txt, docx, pdf).
    """
    filename = file.filename.lower()
    if filename.endswith('.txt'):  # Jika file teks
        return read_txt_file(file)
    elif filename.endswith('.docx'):  # Jika file Word
        return read_docx_file(file)
    elif filename.endswith('.pdf'):  # Jika file PDF
        return read_pdf_file(file)
    return ""  # Jika format tidak dikenal, kembalikan string kosong

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    """
    Melakukan preprocessing pada teks:
    - Case folding
    - Tokenizing
    - Stopword removal
    - Stemming
    """
    text = text.lower()  # Case folding (huruf kecil)
    
    tokens = re.findall(r'\b\w+\b', text)  # Tokenizing (pisahkan kata-kata)
    
    # Daftar stopwords (dapat diperluas sesuai kebutuhan)
    stopwords = set(['yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 'dan', 'atau'])
    tokens = [token for token in tokens if token not in stopwords]  # Filtering
    
    stemmed_tokens = [stem_word(token) for token in tokens]  # Stemming menggunakan kamus
    
    return stemmed_tokens  # Kembalikan token yang sudah diproses

# Fungsi untuk menghitung TF-IDF
def calculate_tf_idf(documents, query):
    """
    Menghitung TF-IDF untuk dokumen dan query.
    """
    doc_tf = [Counter(doc) for doc in documents]  # Hitung term frequency (TF) untuk setiap dokumen
    query_tf = Counter(query)  # Hitung TF untuk query
    
    N = len(documents)  # Jumlah dokumen
    word_set = set().union(*documents)  # Kumpulan semua kata unik
    idf = {}
    
    for word in word_set:  # Iterasi setiap kata
        doc_count = sum(1 for doc in documents if word in doc)  # Hitung dokumen yang mengandung kata
        idf[word] = math.log10(N / doc_count) if doc_count > 0 else 0  # Hitung IDF
    
    # Hitung TF-IDF vectors
    doc_vectors = []
    for tf in doc_tf:
        vector = {word: tf[word] * idf.get(word, 0) for word in word_set}  # TF * IDF
        doc_vectors.append(vector)
    
    query_vector = {word: query_tf[word] * idf.get(word, 0) for word in word_set}  # Query TF-IDF
    
    return doc_vectors, query_vector  # Kembalikan vektor dokumen dan query

# Fungsi untuk menghitung cosine similarity
def cosine_similarity(vec1, vec2):
    """
    Menghitung cosine similarity antara dua vektor.
    """
    intersection = set(vec1.keys()) & set(vec2.keys())  # Kata yang sama pada kedua vektor
    
    numerator = sum([vec1[x] * vec2[x] for x in intersection])  # Hitung pembilang
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])  # Hitung norma vektor pertama
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])  # Hitung norma vektor kedua
    denominator = math.sqrt(sum1) * math.sqrt(sum2)  # Hitung penyebut
    
    if not denominator:
        return 0.0  # Jika penyebut nol, cosine similarity adalah 0
    return float(numerator) / denominator  # Kembalikan cosine similarity

# Route untuk halaman utama
@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Halaman utama untuk upload file dan input query.
    """
    # Variabel untuk menampilkan hasil
    query = None
    original_query = None
    processed_query = None
    similarities = []
    documents_data = []

    if request.method == 'POST':  # Jika form dikirim
        query = request.form['query']  # Ambil query dari form
        
        if 'documents[]' not in request.files:  # Jika tidak ada file
            return render_template('index.html', error='Tidak ada file yang dipilih')
        
        files = request.files.getlist('documents[]')  # Ambil semua file
        
        # Proses setiap file yang diupload
        documents = []
        doc_names = []
        original_tokens = []  # Untuk menyimpan token sebelum preprocessing
        
        for file in files:
            if file and allowed_file(file.filename):  # Jika file valid
                filename = secure_filename(file.filename)  # Amankan nama file
                try:
                    content = read_file_content(file)  # Baca isi file
                    if content:
                        tokens = re.findall(r'\b\w+\b', content.lower())  # Ambil token asli
                        original_tokens.append(tokens)  # Simpan token
                        documents.append(preprocess_text(content))  # Proses preprocessing
                        doc_names.append(filename)  # Simpan nama file
                except Exception as e:
                    print(f"Error membaca file {filename}: {str(e)}")  # Tampilkan error jika ada
                    continue
        
        if not documents:  # Jika tidak ada dokumen valid
            return render_template('index.html', error='Tidak ada file valid yang diupload')
        
        # Preprocess query
        original_query = re.findall(r'\b\w+\b', query.lower())  # Tokenize query asli
        processed_query = preprocess_text(query)  # Preprocessing query
        
        # Hitung TF-IDF
        doc_vectors, query_vector = calculate_tf_idf(documents, processed_query)
        
        # Hitung cosine similarity
        for i, doc_vector in enumerate(doc_vectors):
            sim = cosine_similarity(doc_vector, query_vector)  # Hitung similarity
            similarities.append((doc_names[i], sim))  # Simpan hasil
        
        # Urutkan berdasarkan similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        documents_data = list(zip(doc_names, original_tokens, documents))  # Gabungkan hasil
        
    # Render halaman utama
    return render_template('index.html', 
                           query=query,
                           original_query=original_query,
                           processed_query=processed_query,
                           similarities=similarities,
                           documents=documents_data)

# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)
