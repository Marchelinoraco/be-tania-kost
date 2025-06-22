import { useState } from "react";
import axios from "axios";

function App() {
  const [namaKos, setNamaKos] = useState("");
  const [jumlah, setJumlah] = useState(5);
  const [hasil, setHasil] = useState(null);
  const [error, setError] = useState("");

  const handleCari = async () => {
    try {
      const res = await axios.get(
        `http://localhost:5000/rekomendasi?nama=${encodeURIComponent(
          namaKos
        )}&jumlah=${jumlah}`
      );
      setHasil(res.data);
      setError("");
    } catch (err) {
      setHasil(null);
      setError(err.response?.data?.error || "Terjadi kesalahan.");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-r from-indigo-100 via-white to-pink-100 p-6">
      <div className="max-w-2xl mx-auto bg-white rounded-2xl shadow-xl p-6">
        <h1 className="text-2xl font-bold text-center mb-4 text-indigo-600">
          Sistem Rekomendasi Indekos
        </h1>

        <div className="flex flex-col gap-3">
          <input
            type="text"
            placeholder="Masukkan Nama Indekos"
            value={namaKos}
            onChange={(e) => setNamaKos(e.target.value)}
            className="p-2 border rounded-xl shadow-sm focus:outline-none focus:ring focus:border-indigo-300"
          />

          <input
            type="number"
            min={1}
            max={10}
            value={jumlah}
            onChange={(e) => setJumlah(e.target.value)}
            className="p-2 border rounded-xl shadow-sm focus:outline-none focus:ring focus:border-indigo-300"
          />

          <button
            onClick={handleCari}
            className="bg-indigo-600 text-white px-4 py-2 rounded-xl hover:bg-indigo-700 transition-all"
          >
            Cari Rekomendasi
          </button>
        </div>

        {error && <p className="mt-4 text-red-500 text-sm">{error}</p>}

        {hasil && (
          <div className="mt-6 space-y-6">
            <div className="border rounded-xl p-4 shadow bg-green-50">
              <h2 className="text-xl font-bold text-green-700 mb-2">
                Indekos yang Anda pilih:
              </h2>
              <p>
                <strong>Nama:</strong> {hasil.input}
              </p>
            </div>

            <div>
              <h2 className="text-xl font-semibold mb-2 text-indigo-700">
                Rekomendasi Lainnya:
              </h2>
              <ul className="space-y-4">
                {hasil.hasil.map((item, index) => (
                  <li
                    key={index}
                    className="border rounded-xl p-4 shadow-sm bg-indigo-50"
                  >
                    <h3 className="text-lg font-semibold text-indigo-800">
                      {item.nama}
                    </h3>
                    <p>
                      <strong>Jenis:</strong> {item.jenis}
                    </p>
                    <p>
                      <strong>Harga:</strong> Rp{" "}
                      {item.harga.toLocaleString("id-ID")}
                    </p>
                    <p>
                      <strong>Fasilitas:</strong> {item.fasilitas}
                    </p>
                    <p>
                      <strong>Jarak:</strong> {item.jarak} meter
                    </p>
                    <p>
                      <strong>Skor Kemiripan:</strong> {item.skor_kemiripan}
                    </p>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
