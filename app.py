from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from math import log10, sqrt
import numpy as np


# ===================== DATA ===================== #
data = pd.DataFrame({
    "Nama Indekos": [
        "Kost Genteng Biru", "Kost Genteng Merah", "Kost Nibil", "Kost Romancy",
        "Kost Bonita", "Kost Glory", "Kost Executive Family", "Kost Anugerah",
        "Kos Rajawali", "Kost Mulia", "D’Kost"
    ],
    "Jenis Indekos": [
        "Putri", "Putri", "Putri", "Putri",
        "Putri", "Campur", "Campur", "Campur",
        "Campur", "Campur", "Campur"
    ],
    "Harga": [
        800000, 800000, 750000, 1200000,
        700000, 900000, 850000, 1000000,
        1200000, 1000000, 1250000
    ],
    "Fasilitas": [
        "Wifi, Kamar Mandi Dalam, Listrik, Dapur Bersama",
        "Wifi, Kamar Mandi Dalam, Kulkas Bersama, Listrik",
        "Lemari, Meja Belajar, Listrik, Kamar Mandi Dalam, Dapur Bersama",
        "Wifi, Meja Belajar, Meja Rias, Tempat Tidur, Lemari, Kamar Mandi Dalam, Kulkas Bersama, Dapur Bersama, AC",
        "Wifi, Kamar Mandi Dalam",
        "Wifi, Kamar Mandi Dalam",
        "Wifi, Kamar Mandi Dalam, Dapur Bersama, Listrik, Parkiran",
        "Wifi, Kamar Mandi Dalam, AC, Dapur Bersama, Parkiran",
        "Wifi, Kamar Mandi Dalam, Dapur Bersama, AC, Parkiran",
        "Wifi, Kamar Mandi Dalam, Dapur Bersama, AC, Parkiran",
        "Wifi, Kamar Mandi Dalam, AC, Parkiran"
    ],
    "Jarak": [230, 235, 250, 230, 500, 300, 280, 650, 500, 700, 900],
     "Kontak": [
        "6281234567890", "6281234567891", "6281234567892", 
        "6281234567893", "6281234567894", "6281234567895", 
        "6281234567896", "6281234567897", "6281234567898", 
        "6281234567899", "6281234567800"
    ]
})

# ===================== GABUNG FITUR ===================== #
def gabungkan_fitur(row):
    return f"{row['Nama Indekos']} {row['Jenis Indekos']} harga {row['Harga']} jarak {row['Jarak']} {row['Fasilitas']}"

data["gabungan_fitur"] = data.apply(gabungkan_fitur, axis=1).str.lower()
nama_indekos = data["Nama Indekos"]

# ===================== TF-IDF SIMILARITY ===================== #
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data["gabungan_fitur"])
global_similarity = cosine_similarity(tfidf_matrix)

# ===================== FLASK APP ===================== #
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({"message": "Selamat datang di API Rekomendasi Kost (TF-IDF All Fitur)."})

@app.route('/semua-kost', methods=['GET'])
def semua_kost():
    hasil = []
    for i in range(len(data)):
        hasil.append({
            "nama": data.iloc[i]["Nama Indekos"],
            "jenis": data.iloc[i]["Jenis Indekos"],
            "harga": int(data.iloc[i]["Harga"]),
            "fasilitas": data.iloc[i]["Fasilitas"],
            "jarak": int(data.iloc[i]["Jarak"]),
        })
    return jsonify({"jumlah": len(hasil), "hasil": hasil})


@app.route('/manual-rekomendasi-detail', methods=['GET'])
def manual_rekomendasi_detail():
    # ================== Step 1: Definisi Fitur ==================
    fitur = [
        "Wifi", "Kamar Mandi Dalam", "Listrik", "Dapur Bersama", "Meja Belajar",
        "Meja Rias", "Tempat Tidur", "Lemari", "Kulkas Bersama", "AC", "Parkiran"
    ]

    log = {"steps": []}

    # ================== Step 2: TF ==================
    tf = []
    for fasilitas in data["Fasilitas"]:
        tokens = fasilitas.lower()
        row = [1 if f.lower() in tokens else 0 for f in fitur]
        tf.append(row)
    tf = np.array(tf)

    tf_list = tf.tolist()
    log["steps"].append({"title": "TF (Term Frequency)", "data": tf_list, "fitur": fitur})

    # ================== Step 3: IDF ==================
    df = tf.sum(axis=0)
    N = len(data)
    idf = [round(log10(N / (df[i] + 1e-10)), 3) for i in range(len(fitur))]

    log["steps"].append({"title": "IDF", "data": idf, "fitur": fitur})

    # ================== Step 4: TF-IDF ==================
    tfidf = tf * np.array(idf)
    tfidf_list = tfidf.tolist()
    log["steps"].append({"title": "TF-IDF", "data": tfidf_list, "fitur": fitur})

    # ================== Step 5: Harga & Jarak Normalisasi ==================
    harga = data["Harga"].astype(float)
    jarak = data["Jarak"].astype(float) / 1000  # m -> km

    harga_min, harga_max = harga.min(), harga.max()
    jarak_min, jarak_max = jarak.min(), jarak.max()

    skor_harga = ((harga_max - harga) / (harga_max - harga_min + 1e-10)).round(3)
    skor_jarak = ((jarak_max - jarak) / (jarak_max - jarak_min + 1e-10)).round(3)

    log["steps"].append({
        "title": "Normalisasi Harga",
        "data": skor_harga.tolist()
    })
    log["steps"].append({
        "title": "Normalisasi Jarak",
        "data": skor_jarak.tolist()
    })

    # ================== Step 6: Jenis ==================
    jenis = data["Jenis Indekos"].apply(lambda x: 1 if "putri" in x.lower() else 0)
    log["steps"].append({
        "title": "Jenis (Putri=1, Campur=0)",
        "data": jenis.tolist()
    })

    # ================== Step 7: Gabungkan Semua Vektor Indekos ==================
    fitur_unik = tfidf.sum(axis=1)
    final_matrix = np.vstack([fitur_unik, skor_harga, skor_jarak, jenis]).T

    log["steps"].append({
        "title": "Vektor Akhir Setiap Indekos",
        "columns": ["TF-IDF Unik", "Skor Harga", "Skor Jarak", "Jenis"],
        "data": final_matrix.tolist(),
        "nama_kost": data["Nama Indekos"].tolist()
    })

    # ================== Step 8: Vektor Preferensi User ==================
    idx_ac = fitur.index("AC")
    user_vector = np.array([
        idf[idx_ac],  # TF-IDF AC
        1.0,          # harga
        1.0,          # jarak
        1.0           # jenis
    ])
    log["steps"].append({
        "title": "Vektor Preferensi User",
        "data": user_vector.tolist(),
        "keterangan": ["AC", "Harga Prefer Murah", "Jarak Dekat", "Jenis Putri"]
    })

    # ================== Step 9: Cosine Similarity ==================
    def cosine_sim(a, b):
        return np.dot(a, b) / (sqrt(np.dot(a, a)) * sqrt(np.dot(b, b)) + 1e-10)

    similarities = []
    for i in range(len(data)):
        sim = cosine_sim(final_matrix[i], user_vector)
        similarities.append((i, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    ranking = []
    for idx, sim in similarities:
        kost = data.iloc[idx]
        ranking.append({
            "nama": kost["Nama Indekos"],
            "vektor": final_matrix[idx].tolist(),
            "skor_kemiripan": round(float(sim), 3)
        })

    log["steps"].append({
        "title": "Perhitungan Cosine Similarity & Ranking",
        "data": ranking
    })

    return jsonify(log)

@app.route('/search', methods=['GET'])
def search():
    q = request.args.get('q', '').lower()
    jumlah = int(request.args.get('jumlah', 5))

    if not q.strip():
        return jsonify({"error": "Query kosong"}), 400

    # Tambahkan query ke akhir data
    combined = data["gabungan_fitur"].tolist() + [q]

    # TF-IDF: Fit ke semua data + query
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined)

    # Ambil skor kemiripan antara query (baris terakhir) dan semua data
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Urutkan dari skor tertinggi
    ranked_indices = similarity_scores.argsort()[::-1]

    hasil = []
    for i in ranked_indices[:jumlah]:
        row = data.iloc[i]
        hasil.append({
            "nama": row["Nama Indekos"],
            "jenis": row["Jenis Indekos"],
            "harga": int(row["Harga"]),
            "fasilitas": row["Fasilitas"],
            "jarak": int(row["Jarak"]),
            "skor_kemiripan": round(float(similarity_scores[i]), 3)
        })

    return jsonify({
        "query": q,
        "hasil": hasil
    })


@app.route('/search-detail', methods=['GET'])
def search_detail():
    q = request.args.get('q', '').lower()
    jumlah = int(request.args.get('jumlah', 5))

    if not q.strip():
        return jsonify({"error": "Query kosong"}), 400

    # Gabungan fitur + query user
    combined = data["gabungan_fitur"].tolist() + [q]

    # Fit dan transform
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined)

    # Fitur (kata-kata) dan indeks
    fitur = vectorizer.get_feature_names_out()
    query_vector = tfidf_matrix[-1].toarray().flatten()
    data_vectors = tfidf_matrix[:-1].toarray()

    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    ranked = similarity_scores.argsort()[::-1]

    hasil = []
    for i in ranked[:jumlah]:
        row = data.iloc[i]
        item_vector = data_vectors[i]
        
        # Detail TF-IDF per fitur (untuk query dan item)
        tfidf_detail = []
        for j, kata in enumerate(fitur):
            tfidf_detail.append({
                "fitur": kata,
                "tfidf_query": round(query_vector[j], 4),
                "tfidf_item": round(item_vector[j], 4),
                "produk_tfidf": round(query_vector[j] * item_vector[j], 4)
            })

        # Cosine similarity manual
        dot_product = float(np.dot(query_vector, item_vector))
        norm_query = float(np.linalg.norm(query_vector))
        norm_item = float(np.linalg.norm(item_vector))
        similarity = dot_product / (norm_query * norm_item + 1e-10)

        hasil.append({
            "nama": row["Nama Indekos"],
            "jenis": row["Jenis Indekos"],
            "harga": int(row["Harga"]),
            "fasilitas": row["Fasilitas"],
            "jarak": int(row["Jarak"]),
            "skor_kemiripan": round(similarity, 3),
            "perhitungan": {
                "dot_product": round(dot_product, 4),
                "norm_query": round(norm_query, 4),
                "norm_item": round(norm_item, 4),
                "cosine_similarity": round(similarity, 4),
                "tfidf_detail": tfidf_detail
            }
        })

    hasil_sorted = sorted(hasil, key=lambda x: x["jarak"])
    return jsonify({
    "query": q,
    "hasil": hasil_sorted
    })



@app.route('/rekomendasi', methods=['GET'])
def rekomendasi():
    nama_kos = request.args.get('nama')
    jumlah = int(request.args.get('jumlah', 5))

    if nama_kos not in nama_indekos.values:
        return jsonify({"error": "Nama kost tidak ditemukan."}), 404

    idx = nama_indekos[nama_indekos == nama_kos].index[0]
    skor_kemiripan = list(enumerate(global_similarity[idx]))
    skor_kemiripan = sorted(skor_kemiripan, key=lambda x: x[1], reverse=True)

    hasil = []
    for i, skor in skor_kemiripan[1:jumlah + 1]:
        row = data.iloc[i]
        hasil.append({
            "nama": row["Nama Indekos"],
            "jenis": row["Jenis Indekos"],
            "harga": int(row["Harga"]),
            "fasilitas": row["Fasilitas"],
            "jarak": int(row["Jarak"]),
            "skor_kemiripan": round(float(skor), 3)
        })

    return jsonify({
        "input": nama_kos,
        "jumlah_rekomendasi": jumlah,
        "hasil": hasil
    })

@app.route('/search-full', methods=['GET'])
def search_full():
    q = request.args.get('q', '').lower()
    jumlah = int(request.args.get('jumlah', 5))

    if not q.strip():
        return jsonify({"error": "Query kosong"}), 400

    # Gabungkan query user dengan data
    combined = data["gabungan_fitur"].tolist() + [q]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined)

    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    ranked = similarity_scores.argsort()[::-1]

    hasil = []
    for i in ranked[:jumlah]:
        row = data.iloc[i]
        hasil.append({
            "nama": row["Nama Indekos"],
            "jenis": row["Jenis Indekos"],
            "harga": int(row["Harga"]),
            "fasilitas": row["Fasilitas"],
            "jarak": int(row["Jarak"]),
            "skor_kemiripan": round(float(similarity_scores[i]), 3)
        })

    return jsonify({
        "query": q,
        "hasil": hasil
    })


@app.route('/detail-kost')
def detail_kost():
    nama = request.args.get('nama')
    for i in range(len(data)):
        if data.iloc[i]['Nama Indekos'].lower() == nama.lower():
            kost = {
                "nama": data.iloc[i]["Nama Indekos"],
                "jenis": data.iloc[i]["Jenis Indekos"],
                "harga": int(data.iloc[i]["Harga"]),
                "fasilitas": data.iloc[i]["Fasilitas"],
                "jarak": int(data.iloc[i]["Jarak"]),
                "kontak": data.iloc[i]["Kontak"]
            }
            return jsonify({"kost": kost})
    return jsonify({'error': 'Kost tidak ditemukan'}), 404


# ===================== RUN ===================== #
if __name__ == '__main__':
    app.run(debug=True, port=5001)
