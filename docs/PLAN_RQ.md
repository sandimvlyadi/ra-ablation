# Rencana arm per RQ (disetujui pengguna 2026-11-10/11; belum ada run)

**Kolom:**
- **Jenis:** *rujukan*, *metode ada* (perbandingan sistematis), *modifikasi* (kebaruan; selalu dibandingkan dengan **versi asli** dan **metode terbaik**), *pasca-latih* (tanpa training).
- **Program:** *flag ada* = cukup flag `train.py` yang sudah ada; *kecil* = flag baru dengan sedikit kode; *baru* = modul baru.
- **Prio:** A = jalan duluan; B = setelah A, bila waktu cukup.
- **Jam:** jam GPU kasar dari waktu E3. 1 arm = 6 run (lihat `KONFIGURASI.md`). ResNet34 ≈ 1,9 jam/arm. ViT-S ≈ 2,9 (asumsi). Pre-training SSL baru ≈ 6–8 jam (dari E1).
- Sumber metode: kode M… atau C… = studi SLR-M (`sources/slr-m_kandidat_ablasi.md`). "Umum" = literatur umum di luar SLR-M; sitasi paper aslinya diverifikasi sebelum dipakai.
- Kebaruan setiap modifikasi dicek lewat literatur sebelum diklaim.

## RQ0: rujukan bersama
| Kode | Metode | Jenis | Pembanding | Program | Prio | Jam |
|---|---|---|---|---|---|---|
| RQ0-01 | ResNet34 + ImageNet + B2 (balanced softmax), aug `base` | rujukan | — | flag ada | A | 1,9 |
| RQ0-02 | ConvNeXt-Tiny + ImageNet + B2 (cek arsitektur; C093, C015, M157) | metode ada | RQ0-01 | flag ada (`--model convnext_tiny`) | A | 2,5 |

## RQ1: SSL dan pre-training (kepala tetap B2)
| Kode | Metode | Jenis | Pembanding | Program | Prio | Jam |
|---|---|---|---|---|---|---|
| RQ1-01 | Scratch ResNet34 | metode ada | RQ0-01 | flag ada | A | 1,9 |
| RQ1-02 | DINO ResNet34 (artefak E1) | metode ada | RQ0-01 | flag ada (`--init-from`) | A | 1,9 |
| RQ1-03 | DINO + ChronoCon ResNet34 (artefak E1) | metode ada | RQ1-02 | flag ada | A | 1,9 |
| RQ1-04 | ImageNet ViT-S + LLRD | metode ada | RQ0-01 | flag ada | A | 2,9 |
| RQ1-05 | MAE ViT-S + LLRD (artefak E1) | metode ada | RQ1-04 | flag ada | A | 2,9 |
| RQ1-06 | DINO ViT-S + LLRD (artefak E1) | metode ada | RQ1-04 | flag ada | A | 2,9 |
| RQ1-07 | Pre-training antara: regresi bone age RSNA (tangan), lalu fine-tune (M005) | metode ada | RQ0-01 | baru (+ data Kaggle) | A | 2,9 |
| RQ1-08 | SimCLR ResNet34 (umum) | metode ada | RQ0-01, RQ1-02 | baru (`ssl.py`) | A | ±9 |
| RQ1-09 | MoCo v3 ResNet34 (umum) | metode ada | RQ0-01, RQ1-02 | baru | B | ±9 |
| RQ1-10 | BYOL ResNet34 (umum) | metode ada | RQ0-01, RQ1-02 | baru | A | ±9 |
| RQ1-11 | iBOT ViT-S, gabungan MAE + DINO (umum) | metode ada | RQ1-05, RQ1-06 | baru | B | ±11 |
| RQ1-12 | SupCon + CE (M082, M166) | metode ada | RQ0-01 | baru (loss) | A | 2,5 |
| RQ1-13 | **SupCon berbobot jarak grade** (sudah ada di M219; dilaporkan sebagai adaptasi, tanpa klaim baru, keputusan 2026-11-13) | **modifikasi** | RQ1-12 (asli), terbaik RQ1 | kecil di atas RQ1-12 | A | 2,5 |
| RQ1-14 | SSL normalitas (sendi grade 0) gaya SS-FewSOME, skor kontinu (M197) | metode ada | RQ0-01 | baru | A | 3,9 |
| RQ1-15 | **Skor normalitas sebagai gate grade 0 vs ≥1 di classifier** | **modifikasi** | RQ1-14 (asli), terbaik RQ1 | kecil | A | 1,9 |
| RQ1-16 | **Penyelarasan progresi antarkunjungan (MMD) sebagai ganti ChronoCon** (M335) | **modifikasi** | RQ1-03 (ChronoCon), terbaik RQ1 | baru | B | ±3 |
| RQ1-17 | Strategi fine-tuning: linear probe, LP-FT × {ImageNet, SSL terbaik} | metode ada | fine-tune penuh | flag ada | A | 7,6 |
| RQ1-18 | SKELEX ViT-L, linear probe + fine-tune LLRD (M015) | metode ada | RQ0-01, terbaik RQ1 | baru (loader) | A | **ukur 1 epoch dulu** |
| RQ1-19 | RAD-DINO / BiomedCLIP di-fine-tune (M157) | metode ada | RQ0-01 | baru (loader) | B | ±6 |

## RQ2: imbalance dan ordinalitas (encoder tetap ImageNet ResNet34)
| Kode | Metode | Jenis | Pembanding | Program | Prio | Jam |
|---|---|---|---|---|---|---|
| RQ2-01 | CE polos (`--logit-adjust 0`) | metode ada (baseline sumbu) | — | flag ada | A | 1,9 |
| RQ2-02 | B1: class weight + oversampling, kepala ordinal BCE | metode ada | RQ2-01 | flag ada | A | 1,9 |
| RQ2-03 | Oversampling saja | metode ada | RQ2-01 | flag ada | A | 1,9 |
| RQ2-04 | Class weight saja | metode ada | RQ2-01 | flag ada | A | 1,9 |
| RQ2-05 | B3: CORN | metode ada | RQ2-01 | flag ada | A | 1,9 |
| RQ2-06 | **B4: CORN + logit adjustment** (modifikasi tesis, diulang) | **modifikasi** | RQ2-05 (asli), RQ0-01 | flag ada | A | 1,9 |
| RQ2-07 | B6: decoupled cRT | metode ada | RQ2-01 | flag ada | A | 1,9 |
| RQ2-08 | Undersampling grade 0 berimbang per jenis sendi (E7 `rotate`) | modifikasi lama | RQ2-01 | flag ada | A | 1,9 |
| RQ2-09 | Focal loss γ = 2 | metode ada | RQ2-01 | kecil | A | 1,9 |
| RQ2-10 | Focal + oversampling (M041) | metode ada | RQ2-09 | kecil | A | 1,9 |
| RQ2-11 | Focal ordinal loss (M132) | metode ada | RQ2-09 | kecil | B | 1,9 |
| RQ2-12 | Kepala cumulative link / cut-point (M089) | metode ada | RQ2-01, RQ2-05 | kecil | A | 1,9 |
| RQ2-13 | Soft label ordinal beta/Gaussian (M266, C047) | metode ada | RQ2-01 | kecil | A | 1,9 |
| RQ2-14 | LDAM-DRW (umum) | metode ada | RQ0-01 | kecil | A | 1,9 |
| RQ2-15 | Class-balanced loss, effective number (umum) | metode ada | RQ2-04 | kecil | A | 1,9 |
| RQ2-16 | **B2 + kepala CORN sebagai regularizer** (mirip C015, cek kebaruan) | **modifikasi** | RQ0-01 (asli), RQ2-05 | kecil | B | 1,9 |
| RQ2-17 | **Logit adjustment dengan prior per jenis sendi** | **modifikasi** | RQ0-01 (asli) | kecil | A | 1,9 |
| RQ2-18 | Interaksi: 2 arm terbaik RQ2 pada encoder terbaik RQ1 | interaksi | arm yang sama pada ImageNet | — | A | ±5 |
| RQ2-P | Ambang per kelas disetel di val; kNN di atas embedding (M161); TTA (C093) | pasca-latih | arm asalnya | analisis | A | 0 |

## RQ3: ketahanan RAM-H1200 → RheumaVIT (untuk hasil RQ1 dan RQ2)
| Kode | Metode | Jenis | Pembanding | Program | Prio | Jam |
|---|---|---|---|---|---|---|
| RQ3-01 | Transfer langsung semua run RQ0–RQ2; drop = internal − eksternal | evaluasi | — | baru (`rq3.py`, pakai fungsi `e4.py`) | A | ±1 |
| RQ3-02 | Test-time BN adaptation tanpa label (M095) | metode ada | RQ3-01 | analisis | A | ±0,2 |
| RQ3-03 | Harmonisasi intensitas saat inferensi, histogram matching ke RAM-H1200 (umum) | metode ada | RQ3-01 | analisis | A | ±0,2 |
| RQ3-04 | Augmentasi intensitas kuat / MixStyle saat latih (umum) | metode ada | RQ0-01 | kecil | A | 1,9 |
| RQ3-05 | **Augmentasi intensitas disesuaikan dengan statistik RheumaVIT tanpa label** (crop RheumaVIT lebih terang dan kontras lebih rendah) | **modifikasi** (domain adaptation, dilaporkan terpisah) | RQ3-04 (asli), RQ0-01 | kecil | A | 1,9 |
| RQ3-06 | Few-shot 5/20 pasien untuk arm terbaik RQ1 dan RQ2 (resep E4) | evaluasi | transfer langsung | fungsi `e4.py` | A | ±3 |
| RQ3-07 | Jangkar pembaca: model vs 3 radiolog | evaluasi | — | fungsi `e4.py` | A | 0 |

## RQ4: kalibrasi dan rujukan selektif (di atas RQ0-01 dan arm terbaik RQ2; internal + RheumaVIT)
| Kode | Metode | Jenis | Pembanding | Program | Prio | Jam |
|---|---|---|---|---|---|---|
| RQ4-P1 | Temperature scaling; + log-prior (modifikasi lama, dihitung ulang) | metode ada / modifikasi lama | tanpa kalibrasi | fungsi `e5.py` | A | 0 |
| RQ4-P2 | Vector / Dirichlet scaling (M095) | metode ada | RQ4-P1 | analisis | A | 0 |
| RQ4-P3 | Conformal ordinal APS, RAPS (M110), Mondrian per grade | metode ada | — | fungsi `e5.py` + baru | A | 0 |
| RQ4-P4 | **Mondrian per jenis sendi × grade** | **modifikasi** | Mondrian per grade (asli) | analisis | A | 0 |
| RQ4-P5 | Deep ensemble antar seed sebagai skor rujukan | metode ada | — | analisis | A | 0 |
| RQ4-P6 | **Skor rujukan gabungan: varians ensemble + ukuran set conformal** | **modifikasi** | RQ4-P3, RQ4-P5 | analisis | A | 0 |
| RQ4-01 | MC dropout di head + kurva risk-coverage (M130, M110) | metode ada | RQ4-P5 | kecil | A | 1,9 |
| RQ4-02 | Evidential deep learning (M270) | metode ada | RQ4-01 | baru | B | 1,9 |

## RQ5: label noise / label correction / teacher-student (encoder dan kepala = RQ0-01)
| Kode | Metode | Jenis | Pembanding | Program | Prio | Jam |
|---|---|---|---|---|---|---|
| RQ5-00 | Prediksi out-of-fold: 5 fold per pasien di dalam train, 100%, s42 | persiapan | — | baru | A | 2,6 |
| RQ5-01 | CleanLab relabel (label train saja) (M272, M117) | metode ada | RQ0-01 | baru (`cleanlab`) | A | 1,9 |
| RQ5-02 | CleanLab prune | metode ada | RQ0-01 | baru | A | 1,9 |
| RQ5-03 | CleanLab bobot turun | metode ada | RQ0-01 | baru | A | 1,9 |
| RQ5-04 | Label smoothing ε = 0,1 (M044, M238) | metode ada | RQ0-01 | kecil | A | 1,9 |
| RQ5-05 | **Soft label yang lebarnya dari ketidaksepakatan pembaca** (ICC pra-konsensus RAM-H1200) | **modifikasi** | RQ5-04 (asli), RQ2-13 | kecil | A | 1,9 |
| RQ5-06 | Pemisahan sampel berdasarkan confidence + loss hibrida (M200) | metode ada | RQ0-01 | baru | A | 1,9 |
| RQ5-07 | **Reliabilitas dari konsistensi antarkunjungan** (grade 1→0: 47%) | **modifikasi** | RQ5-06 (asli), terbaik RQ5 | kecil di atas RQ5-06 | A | 1,9 |
| RQ5-08 | GCE, generalized cross-entropy (umum) | metode ada | RQ0-01 | kecil | A | 1,9 |
| RQ5-09 | Co-teaching (M095; umum) | metode ada | RQ0-01 | baru | B | 3,8 |
| RQ5-10 | DivideMix (umum) | metode ada | RQ0-01 | baru | B | 5,7 |
| RQ5-11 | Mean teacher / consistency semi-supervised, 10% dan 25% (M061, M185) | metode ada | RQ0-01 pada fraksi sama | baru | A | ±4 |
| RQ5-12 | Self-distillation dari ensemble seed (M155, M095) | metode ada | RQ0-01 | baru | B | 1,9 |
| RQ5-E | Evaluasi RheumaVIT per pembaca dan terhadap distribusi pembaca | evaluasi | — | analisis | A | 0 |

## RQ6: kombinasi terbaik
| Kode | Metode | Jenis | Pembanding | Program | Prio | Jam |
|---|---|---|---|---|---|---|
| RQ6-01 | Encoder terbaik + kepala terbaik + metode label terbaik + kalibrasi terbaik, internal + RheumaVIT | kombinasi | RQ0-01 | — | A | ±3 |

## Total (kasar)
| RQ | Jam |
|---|---|
| RQ0 | ±4 |
| RQ1 | ±83 + SKELEX (prioritas A saja ±52) |
| RQ2 | ±37 (A ±33) |
| RQ3 | ±8 |
| RQ4 | ±4 (A ±2) |
| RQ5 | ±35 (A ±24) |
| RQ6 | ±3 |
| **Semua** | **±175 jam** (~22 malam GPU) |
| **Prioritas A saja** | **±125 jam** (~16 malam) |

Total ini naik dari perkiraan ±148 jam di diskusi karena setiap modifikasi sekarang punya arm versi aslinya.
