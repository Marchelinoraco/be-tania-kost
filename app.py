from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

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
