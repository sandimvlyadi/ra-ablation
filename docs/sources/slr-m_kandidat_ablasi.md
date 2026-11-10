# SLR-M: kandidat ablasi per RQ (setelah full text)

**Status:** 2026-11-10. Menggantikan `01_kandidat_ablasi_sementara.md` (berbasis abstrak).
- 50 dari 69 studi dibaca full text-nya, lewat potongan terarah pada bagian dataset, split, metode, ablasi, hasil, kode, dan komputasi. Paper tidak dibaca kata per kata.
- 19 studi tanpa full text diekstrak dari abstrak dan ditandai †.
- Semua keputusan berasal dari satu reviewer AI dan wajib diverifikasi manusia.

**Data:**
- tabel ekstraksi: `extraction/01_fulltext_extraction.tsv`
- screening: `screening/`
- protokol: `protocol/01_protocol_PRISMA-P.md`

## 1. Alur PRISMA

| Tahap | n |
|---|---|
| Teridentifikasi: Scopus 290, PubMed 96, IEEE 41 (dari 310, setelah filter DL + G1–G4) | 427 |
| Duplikat | 86 |
| Di-screen judul/abstrak | 341 |
| Dikeluarkan (EM3 144, EM2 69, EM1 36, EM4 18, EM5 4, EM6 1) | 272 |
| Laporan dicari | 69 |
| Tidak diperoleh (berbayar di luar langganan, atau "free access" yang tetap tidak bisa diunduh) | 19 |
| Dinilai full text | 50 |
| Dikeluarkan di tahap full text | 0 (M173 dipertahankan sebagai bukti latar G2 meski tanpa hasil grading; M283 = laporan ganda M141, dihitung sekali) |
| Studi dari database | 69 (50 full text + 19 hanya abstrak) |
| Citation searching (OpenAlex, 1 putaran dari 38 studi berablasi): 1.157 backward + 417 forward; 95 baru setelah filter tugas + radiografi + DL | 95 di-screen |
| Dikeluarkan (EM3 tanpa metode G1–G4 / EM2, EM4 review/retraksi, EM7 preprint 8) | 91 |
| Masuk dari citation searching (full text semua diperoleh) | 4 (C007, C015, C037, C093) |
| **Studi dalam sintesis** | **73** (54 full text + 19 hanya abstrak) |

Dicatat tetapi tidak eligibel: C047 (preprint arXiv 2026, soft labelling KL dengan 4 bentuk distribusi: binomial, beta, triangular, eksponensial; semuanya mengalahkan baseline nominal). Relevan untuk kandidat 2.3/5.4.

## 2. Kualitas bukti (checklist 6 butir, 50 full text)

| Butir | Ya | Tidak | Tidak jelas | Sebagian |
|---|---|---|---|---|
| Ablasi pada split yang sama | 38 | 5 | 7 | |
| Lebih dari satu run, atau ada CI | 17 | 20 | 13 | |
| Split per pasien | 8 | 13 | 29 | |
| Hiperparameter dipilih di validasi | 6 | | 44 | |
| Uji eksternal / lintas senter | 10 | 35 | 5 | |
| Kode tersedia | 9 | 37 | | 4 |

**Peringatan:**
- **Klaim yang tidak masuk akal**, jauh di atas kesepakatan antarpembaca KL (QWK sekitar 0,56–0,82): M095 (macro-F1 0,982), M150 (QWK 0,99), M175 (> 95%), M188 (mAP ~99%). Studi-studi ini hanya dipakai sebagai "menu" komponen, bukan sebagai bukti besar efek.
- **Label uji ikut diubah:** M042 dan M272. Pada M272, akurasi naik dari 69% ke 82% hanya bila diuji pada label uji yang sudah dibersihkan CleanLab. Pada label uji asli, selisihnya **tidak bermakna (p = 0,076)**. Ini pelajaran utama untuk desain RQ5.
- Literatur didominasi OA lutut, dan banyak studi memakai dataset Kaggle/Mendeley turunan OAI yang sama dengan split per citra.

## 3. Kandidat per RQ

Kolom **Status**: *ada* = sudah diuji; *mirip* = ada varian dekat; *belum* = belum pernah diuji.
Kolom **Prioritas**: A = pertama, B = kedua, C = opsional. Prioritas adalah penilaian reviewer (*reader-inferred*), berdasarkan bukti, biaya, dan kecocokan dengan RAM-H1200.

### RQ1: pre-training dan SSL

Sudah ada di tesis: scratch, ImageNet, MAE, DINO, DINO + ChronoCon, LLRD, linear probe.

| # | Kandidat | Bukti utama | Status | Prioritas | Catatan untuk RAM-H1200 |
|---|---|---|---|---|---|
| 1.1 | **SKELEX**: foundation model MSK, MAE yang dilanjutkan pada 1,3 juta radiograf | M015: rata-rata +6,21% relatif terhadap inisialisasi ImageNet-MAE yang sama (protokol identik); split per pasien; **bobot di GitHub (akademik)** | belum | A | ViT-L berat untuk RTX 3060. Coba linear probe dulu, lalu fine-tune dengan batch kecil + LLRD |
| 1.2 | **Pre-training antara supervised pada radiograf tangan** (RSNA bone age, 14.036 citra) | M005: ImageNet → bone age → SvdH; beberapa skema fine-tuning + ensemble, setara pembaca (PCC 0,925) | belum | A | Data tangan publik, murah; sumbu baru "domain-adjacent supervised" |
| 1.3 | **Supervised contrastive + CE**, termasuk negatif berbobot jarak grade | M082†, M166†, M280† (sadar usia/jenis kelamin), M219 (negatif berbobot jarak KL) | belum | A | (R) Positif = grade sama lintas pasien; bobot negatif ∝ jarak grade; varian sadar jenis sendi |
| 1.4 | **SSL "normalitas"** dari sendi sehat + pseudo label | M197: dilatih dari sendi sehat saja dengan < 3% label, AUC deteksi OA 56,7 → 81,0; 5 seed; **kode ada** | belum | A | Sendi grade 0 sangat banyak; jarak ke pusat normal sebagai skor atau gate |
| 1.5 | **Penyelarasan representasi progresi** (TCN atas sekuens + MMD, inferensi citra tunggal) | M335: model penuh mengalahkan ablasi tanpa temporal pada dua dataset OA | mirip (ChronoCon, E9) | B | Memakai kunjungan tanpa memaksa urutan per sendi, beda dari ChronoCon |
| 1.6 | **Autoencoder adversarial / rekonstruksi + tugas bantu** | M153 (DC-AAE + tugas bantu sisi kaki, data tak berlabel tambahan), M185 | mirip (MAE) | B | (R) Tugas bantu gratis dari ROI: jenis sendi, sisi tangan |
| 1.7 | **Foundation model medis umum** (BiomedCLIP, RAD-DINO) | M157: embedding **beku** + XGBoost kalah dari CNN supervised (ConvNeXt κ 0,880); fine-tune tidak diuji | belum | B | Hanya layak bila di-fine-tune |
| 1.8 | **Vision-language / CLIP** dengan teks templat per grade | M201 (eksternal: akurasi 76,9%), M306 (templat laporan dari skor tabular), M071 (ablasi: ViT-CORAL 65,2 → 70,3% akurasi), M014† | belum | C | Templat SvdH per grade dan per sendi |
| 1.9 | **Strategi fine-tuning**: freeze, gradual unfreeze, LP-FT | M051 (lemah), M269 (gradual unfreezing), M005 | mirip (LLRD, linear probe) | B | Satu sumbu murah pada encoder yang sama |

### RQ2: class imbalance dan ordinalitas

Sudah ada di tesis: B0–B6, E7 undersampling, E8 augmentasi.

| # | Kandidat | Bukti utama | Status | Prioritas | Catatan |
|---|---|---|---|---|---|
| 2.1 | **Oversampling murni** (tanpa class weight) dan **class weight murni** | M041 (OA tangan DIP, 6 strategi): strategi dengan oversampling paling konsisten; SD + OS menaikkan F1 OA 0,416 → 0,613 (p = 0,029); M141/M283: weighted sampling +0,01 akurasi | **belum**: B1 selalu menggabungkan keduanya | A | Memecah B1 menjadi dua arm |
| 2.2 | **Focal loss / focal ordinal loss** | M132 (FOL > CE dan fixed-penalty, 10 ulangan); M041: focal saja **lemah** di KL3/KL4; M100 (erosi RA, RetinaNet) | belum | A | Uji focal sendiri dan dikombinasikan dengan OS (temuan M041) |
| 2.3 | **Kepala ordinal lain**: cumulative link / cut-point, CORAL, penalti jarak, soft label ordinal | M039: CORAL → CE menurunkan QWK 0,068 (p < 0,001); M089 (cut-point pada 23 jaringan); M305; M266† (20 kombinasi, terbaik CE + cumulative link + soft label beta) | mirip (B0, B3, B4); **belum** untuk cumulative link, soft label, penalti jarak | A | Murah: hanya mengganti kepala |
| 2.3b | **Kepala ordinal tambahan sebagai regularizer** (softmax + cumulative ordinal + residual; inferensi pakai head softmax yang dikalibrasi di validasi) | C015: QWK 0,845, akurasi +2,11 poin vs hibrida ResNet34-ViT; tidak ada sampel uji untuk tuning | belum | B | Murah: tambah head CORN/CORAL ke B2 |
| 2.4 | **Ambang keputusan disetel di validasi** | M071 (ambang dioptimalkan di val untuk macro-F1), M052† | belum | A | **Tanpa training baru**: logit tersimpan; ambang per kelas untuk BACC4 |
| 2.5 | **kNN / retrieval-augmented classifier** di atas embedding | M161: EfficientNet-B0 83,5 → 86,8%; tidak naik pada 4 dari 17 arsitektur | mirip (B6) | B | Murah dengan embedding tersimpan |
| 2.6 | **Mixup / DAB-mix / Balanced-MixUp** | M224†, M269 | belum | B | Sudah tercatat sebagai arm lanjutan E8 |
| 2.7 | **Augmentasi lebih kuat untuk grade jarang** | M101†, M275 | belum | B | Gabung dengan resep E8 |
| 2.8 | **Uji shortcut**: hapus atau masking ROI | M249: model tetap benar untuk grade ekstrem walau sendinya dihapus | belum | B (diagnostik) | (R) Masking area lesi atau sendi pada RAM-H1200 |
| 2.9 | **Weighted sampling + class-weighted CE + label smoothing** | M044 (split per pasien, CI, eksternal MOST), M238 | mirip (B1) | C | Selisih per komponen tidak dilaporkan jelas |
| 2.10 | **Data sintetis** (CycleGAN tervalidasi rheumatolog, diffusion, GAN, sintesis lapisan sendi jari) | M041, M068†, M084†, M165, M173, M191, M199 | ditolak sebelumnya | C | Hanya bila ada klinisi validator (seperti M041) |
| 2.10b | **Test-time augmentation** | C093: TTA konsisten menaikkan akurasi; weighted ensemble **tidak** lebih baik dari model tunggal (CI bootstrap) | belum | B | Tanpa training baru |
| 2.11 | SMOTE, ensemble | M162, M233†, M141, M148†, M225† | belum / mirip | C | Bukti lemah |

### RQ3: ketahanan RAM-H1200 → RheumaVIT

Semua varian RQ1, RQ2, RQ4, dan RQ5 dievaluasi dengan transfer langsung dan few-shot (seperti E4). Pembanding dari literatur:
- **M219**: menggabungkan data target berlabel menaikkan QWK +0,21 dibanding source-only. Contrastive domain adaptation menaikkan lagi QWK 0,678 → 0,732. Relevan untuk arm few-shot 5 dan 20 pasien.
- **M201**: evaluasi internal saja biasanya menggelembungkan hasil 10–15%. **M044**: akurasi turun 0,939 → 0,862 di MOST. Keduanya angka acuan *domain drop*.
- **Test-time BN adaptation** (M095): murah dan tidak butuh label RheumaVIT. Prioritas B.

### RQ4: kalibrasi, uncertainty, rujukan selektif

Sudah ada di tesis: temperature scaling, log-prior, conformal ordinal APS.

| # | Kandidat | Bukti | Status | Prioritas | Catatan |
|---|---|---|---|---|---|
| 4.1 | **RAPS dan varian conformal** (class-conditional / Mondrian per grade) | M110: RAPS coverage 0,91–0,93 pada target 0,90; perlu split kalibrasi | mirip (APS) | A | Tanpa training baru |
| 4.2 | **MC dropout** + kurva risk-coverage | M130: variance MC dropout memprediksi kesalahan, divalidasi eksternal, 15 ulangan; M110; M039 (gating, −0,040 QWK bila dibuang); M280† | belum | A | Perlu dropout di head |
| 4.3 | **Deep ensemble antar seed** | Tidak ada studi SLR-M yang menguji ensemble untuk UQ | belum | A | Tanpa training baru (seed sudah ada) |
| 4.4 | **Evidential deep learning** | M110, M270: akurasi setara, uncertainty memisahkan prediksi benar dan salah | belum | B | — |
| 4.5 | **Dirichlet / vector scaling**, test-time BN sebelum conformal | M095 | mirip (TS) | B | Murah |

### RQ5 (baru): label noise, label correction, teacher-student

Sudah ada di tesis: hanya E9 (penalti pasangan kunjungan, tidak ada gain).

| # | Kandidat | Bukti | Status | Prioritas | Catatan |
|---|---|---|---|---|---|
| 5.1 | **Confident learning / CleanLab**: relabel vs prune vs bobot turun | M272, M117, M042 | belum | A | **Label uji tidak boleh diubah.** M272 hanya menang pada label uji yang ikut dibersihkan. Evaluasi juga di RheumaVIT |
| 5.2 | **Semi-supervised consistency / teacher-student** pada sendi tak berlabel | M061 (konsistensi + active learning, ~10% label, uji di situs terpisah, 5 ulangan), M077, M185 (teacher-student hierarkis, **kode ada**), M153 | belum (celah tercatat 2026-11-09) | A | Paling cocok untuk rezim 10% label |
| 5.3 | **Pemisahan sampel menurut reliabilitas + loss hibrida** | M200† (JBHI 2022), M042 | belum | A | (R) Reliabilitas diturunkan dari konsistensi antarkunjungan (grade 1→0 47%) |
| 5.4 | **Label smoothing / soft label ordinal** | M044, M238, M266† | belum | A | (R) Lebar soft label disesuaikan dengan ketidaksepakatan pembaca |
| 5.5 | **Co-teaching** | M095 (klaim tidak masuk akal, menu saja) | belum | B | ResNet34 × ViT-S |
| 5.6 | **Self-distillation / KD** | M155 (tanpa ablasi), M095 (SWA-EMA), M175 | belum | B | (R) Teacher = ensemble seed |
| 5.7 | **Pseudo label + denoising** | M197 (CLIP denoise) | belum | B | — |
| 5.8 | **Evaluasi terhadap distribusi pembaca** | tidak ada di SLR-M | — | A (desain evaluasi) | RheumaVIT punya 3 radiolog: evaluasi dengan soft label pembaca |

## 4. Ringkasan usulan ke dosen

1. **Tanpa training baru** (memakai logit/embedding tersimpan): 2.4 ambang validasi, 2.5 kNN, 4.1 RAPS/Mondrian, 4.3 deep ensemble.
Catatan: C007 = Tan et al. 2023 (RA2-DREAM, encoding ordinal + undersampling grade 0), yang desainnya sudah dicakup B0 dan E7.
2. **Ablasi utama per RQ** (prioritas A):
   - RQ1: SKELEX, bone-age pre-training, SupCon berbobot jarak, SSL normalitas.
   - RQ2: memecah B1 (OS saja / CW saja), focal ± OS, kepala cumulative link + soft label.
   - RQ4: MC dropout, deep ensemble, RAPS.
   - RQ5: CleanLab (label uji tidak diubah), semi-supervised teacher-student pada 10%, pemisahan reliabilitas, soft label.
3. **Semua varian** dilatih di RAM-H1200 dan diuji di RheumaVIT (RQ3), dengan arm few-shot seperti E4.
4. **Celah literatur yang bisa diangkat sebagai kebaruan:**
   - Tidak ada studi yang mengablasi beberapa metode SSL pada grading erosi per sendi.
   - Tidak ada studi RA yang menangani label noise.
   - Hanya 8 dari 50 studi memakai split per pasien.
   - Tidak ada studi yang memakai ensemble untuk rujukan selektif.

## 5. Keterbatasan

- Reviewer AI tunggal.
- Ekstraksi berbasis potongan terarah, bukan pembacaan penuh. Beberapa angka ablasi belum terambil ("numbers not extracted" di tabel).
- 19 laporan tidak diperoleh.
- Citation searching hanya satu putaran, dan 7 kandidatnya tanpa abstrak sehingga dinilai dari judul saja.
- IEEE disaring setelah pengambilan.
- Publication bias positif pada paper metode.
