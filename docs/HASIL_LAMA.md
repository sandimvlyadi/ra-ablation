# Hasil lama yang relevan (repo slr-ra-experiments, sebagai konteks)

Angka diringkas dari laporan `runs/*_analysis/report.md` dan proposal tesis di repo lama. Cek ulang ke sumbernya sebelum dikutip. Hasil-hasil ini **tidak** dipakai sebagai pembanding statistik di sini, karena semua pembanding dilatih ulang.

| Topik | Temuan |
|---|---|
| RQ1, linear probe beku 10% (E1) | SSL > ImageNet +5,1 s.d. +6,7 poin BACC4. Hanya DINO ResNet34 yang bertahan Holm (+0,067, p 0,001). ChronoCon menurunkan probe 100% (−0,045; −0,076) |
| RQ1, fine-tuning penuh (E2) | SSL > scratch pada sebagian kontras 10%. Tidak ada yang mengalahkan ImageNet |
| RQ1, MAE + LLRD (E6) | Naik +0,040 s.d. +0,102 dibanding resep E2. Tidak terbukti > ImageNet pada 10% (+0,029 [−0,008, +0,073], Holm 0,296) |
| RQ2 (Faktor B awal, ResNet34 scratch 100%) | **Modifikasi B4 (CORN + logit adjustment) vs CORN asli: +0,036 [+0,010, +0,062], p 0,007.** B2 − B0 +0,045 (Holm 0,275) |
| RQ2 (E3 faktorial) | H2 tidak didukung (B4 − B1 −0,036 s.d. +0,007). Interaksi DINO berlawanan (−0,043, Holm 0,041). B2 dan B6 > B0 di banyak sel. B1 over-calling pada label rendah (recall ≥2 0,98–0,99, QWK ≈ 0) |
| RQ2 undersampling (E7) | **Modifikasi** undersampling grade 0 berimbang per jenis sendi > B0 di 3 dari 4 sel, tetapi tidak pernah > B2 |
| Augmentasi (E8) | `none` pada 10%: −0,047 (Holm 0,012, "kecil"). Arm lain tidak terbukti. Overfit tetap ada (train 0,931 → val 0,416 pada 10%) |
| Longitudinal (E9) | **Modifikasi** penalti monoton pasangan kunjungan: tidak ada gain di 100% maupun 10% |
| RQ3 (E4) | SSL mengurangi drop terhadap scratch, tetapi tidak terhadap ImageNet. Model jauh di bawah pembaca manusia |
| RQ4 (E5) | **Modifikasi** log-prior sebelum temperature scaling: ECE kelas 0 turun, tetapi cakupan internal turun ke 0,859–0,867 |
| Diagnosis 2026-11-09 | Label antarkunjungan tidak stabil (grade 1→0: 47%). Baseline tanpa citra memberi test BACC4 0,323 (100%) / 0,295 (10%), bukan 0,25 |

**Pola yang ditemukan:** modifikasi kadang mengalahkan versi aslinya, tetapi belum ada yang mengalahkan metode terbaik yang sudah ada (balanced softmax).
