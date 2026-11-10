# RQ0: pra-spesifikasi

Ditulis 2026-11-11, **sebelum run pertama RQ0**. Berlaku setelah di-commit. Ini commit bertanggal di repo privat, bukan registrasi publik.

**RQ0 bersifat *post hoc*.** Test RAM-H1200 sudah dipakai di repo lama sejak E2, jadi hasilnya tidak boleh dipakai untuk memilih konfigurasi tesis. Pemilihan hanya lewat validasi. RheumaVIT tidak dievaluasi di tahap ini (masuk RQ3-01).

## Pertanyaan
1. **RQ0-01 (rujukan bersama):** apakah resep rujukan, yang dilatih ulang di repo ini, mereproduksi arm `base` E8 di repo lama? Arm ini menjadi pembanding untuk semua RQ.
2. **RQ0-02 (cek arsitektur):** apakah ConvNeXt-Tiny memberi BACC4 yang berbeda dari ResNet34 pada resep yang sama? Sumber: C093, C015, M157 (`docs/sources/slr-m_kandidat_ablasi.md`).

## Desain
Resep bersama ada di `docs/KONFIGURASI.md`:
- ImageNet + kepala softmax + logit adjustment τ = 1 (balanced softmax)
- AMP, AdamW, lr 1e-4, weight decay 1e-3, cosine
- 50 epoch dan minimal 2.000 step
- input 224, batch 32
- augmentasi `base` di-seed (`--seed-aug`)
- checkpoint dipilih dengan QWK validasi (`--monitor qwk`)

Validasi dan test tidak diaugmentasi.

| Arm | Backbone (timm) | Bobot awal | Satu-satunya perbedaan dari RQ0-01 |
|---|---|---|---|
| RQ0-01 | `resnet34` | `resnet34.a1_in1k` (ImageNet-1k) | — |
| RQ0-02 | `convnext_tiny` | `convnext_tiny.fb_in1k` (ImageNet-1k; safetensors sha256 `08b9dc9c3a3a29421de7996761e176501896d1ae7fc3085cf56a643772329276`) | backbone |

Bobot dipilih pengguna pada 2026-11-11: `fb_in1k`, bukan default timm `in12k_ft_in1k`. Alasannya, data pre-training-nya sama dengan ResNet34 (ImageNet-1k), sehingga yang berbeda hanya arsitektur.

Setiap arm dijalankan pada 6 desain, sehingga total 12 run:
- 100% label × seed 42, 1337, 2026
- 10% label × draw pasien 0, 1, 2 pada seed 42

Draw dan seed sama dengan E3/E8. Hiperparameter ConvNeXt-Tiny **tidak** disetel ulang: lr, weight decay, dan jadwal sama dengan ResNet34. Jadi RQ0-02 menguji backbone pada resep rujukan, bukan ConvNeXt yang disetel optimal. Batasan ini dicatat di laporan.

Nama run: `rq0-01_resnet34_b2_f{frac}_d{draw}_s{seed}` dan `rq0-02_convnext_tiny_b2_f{frac}_d{draw}_s{seed}`. Antrean: `scripts/run_rq0.sh`.

## Kontras
- Ukuran sampel (3 seed + 3 draw) dan monitor checkpoint (QWK) dikonfirmasi pengguna pada 2026-11-11.
- **Primer (keluarga Holm, 2 kontras):** BACC4 RQ0-02 − RQ0-01 pada 100% dan pada 10% label. Uji: bootstrap pasien berpasangan, 10.000 replikasi, run ikut diambil ulang (`compare.py`, sama dengan `e8.py`/`e6abl.py`).
- **Pendukung (di luar Holm), uji reproduksi:** RQ0-01 vs run E8 `base` dengan draw/seed yang sama (`/data/Smart-X/SLR-RA/experiments/runs/e8/e8_base_a1_resnet34_b2_*`).
  - Kodenya identik (b6e0135) dan augmentasinya di-seed, jadi selisihnya diharapkan nol atau sangat kecil.
  - Selisih non-nol dilaporkan apa adanya, sebagai ukuran nondeterminisme GPU atau lingkungan.
  - Kontras ini tidak menghasilkan putusan.
- **Sekunder (p tidak dikoreksi):**
  - selisih QWK dan ICC skor total tangan
  - recall per grade
  - proporsi grade 0 di antara prediksi
  - metrik validasi dan epoch terpilih
  - waktu per run

## Aturan putusan (BACC4, kontras primer)
- **Bermakna:** Holm p < 0,05 dan |selisih| ≥ 0,05.
- **Kecil:** Holm p < 0,05 dan |selisih| < 0,05.
- **Tidak terbukti:** selain itu. Dengan 3 run per sel, setengah lebar interval sekitar 0,04, sehingga hasil yang tidak signifikan bukan bukti bahwa dua backbone setara.

Semua hasil dilaporkan, apa pun arahnya.

## Konsekuensi yang dikunci sekarang
- **RQ0-01 tetap menjadi rujukan semua RQ, apa pun hasil RQ0-02.** Encoder di RQ2 tetap ImageNet ResNet34 sesuai `docs/PLAN_RQ.md`.
- ConvNeXt-Tiny hanya boleh menjadi kandidat RQ6 kalau BACC4 validasinya (rata-rata 6 run) lebih tinggi daripada RQ0-01. Angka test tidak dipakai untuk keputusan ini.
- Kalau uji reproduksi menunjukkan selisih BACC4 test |Δ| ≥ 0,02 pada run yang sama, RQ1 tidak dimulai sebelum penyebabnya dicari.

## Di luar cakupan
- Backbone lain, penyetelan hiperparameter per backbone, ukuran input lain.
- Kepala, imbalance, atau augmentasi lain (RQ2 / E8).
- RheumaVIT (RQ3).
