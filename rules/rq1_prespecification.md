# RQ1: pra-spesifikasi

Ditulis 2026-11-11, **sebelum run RQ1 mana pun**. Berlaku setelah di-commit. Ini commit bertanggal di repo privat, bukan registrasi publik. Arm dan pembandingnya mengikuti `docs/PLAN_RQ.md`. Menambah arm di luar daftar ini memerlukan izin pengguna dan adendum bertanggal sebelum run arm itu.

**RQ1 bersifat *post hoc*.** Test RAM-H1200 sudah dipakai di repo lama sejak E2, jadi semua pemilihan (resep, "metode terbaik", encoder untuk RQ1-17) hanya memakai validasi. RheumaVIT dievaluasi di RQ3.

## Pertanyaan
Apakah pre-training atau SSL (metode yang sudah ada, dan modifikasi yang diturunkan dari sifat data) mengubah grading erosi per sendi dibandingkan inisialisasi ImageNet, pada 100% dan 10% label?

## Resep bersama
- **Kepala dan loss:** softmax + logit adjustment τ = 1 (balanced softmax), sama dengan rujukan RQ0-01.
- **Augmentasi dan desain:** aug `base` di-seed, monitor QWK validasi, 6 desain per arm (100% × seed 42/1337/2026; 10% × draw 0/1/2 pada seed 42). Total 6 run per arm.
- **Resep ResNet34:** resep RQ0-01 (AdamW, lr 1e-4, wd 1e-3, cosine, 50 epoch dan minimal 2.000 step, tanpa warmup).
- **Resep ViT-S** (`vit_small_patch16_224`): seperti ResNet34, ditambah LLRD 0,85, lr 1e-4, dan warmup 10% epoch.
  - Nilai ini dipilih pada validasi di tahap tuning E6 (repo lama, MAE, 10% d0 s42) dan tidak disetel ulang di sini.
  - ImageNet ViT-S memakai `vit_small_patch16_224.augreg_in1k` (ImageNet-1k; safetensors sha256 `c20a91d93f4757b0f581519f21ad91a875cf041764873a7a5bccc8e36f360dbf`). Ini bukan default timm `augreg_in21k_ft_in1k` yang dipakai di E2/E6.
  - Diputuskan pengguna pada 2026-11-11 (opsi A). Tujuannya, data pre-training ImageNet sama untuk ResNet34, ConvNeXt-Tiny, dan ViT-S, sehingga SSL di kedua backbone dibandingkan dengan baseline yang setara.
  - Akibatnya, angka ViT-S ImageNet di sini tidak sebanding langsung dengan E2/E6. Baseline ImageNet-1k juga lebih lemah daripada in21k, dan ini dinyatakan di laporan.
- **Encoder SSL E1** (`artifacts/encoders/`, sha256 di `artifacts/SHA256SUMS`, dicek sebelum antrean):
  - Pool pre-training berisi ROI BE + JSN dari train **dan val** RAM-H1200 tanpa label. Pasien test dikeluarkan (`pool.py`).
  - Pada rezim 10%, pool ini memuat pasien yang labelnya tidak dipakai. Itulah skenario semi-supervised yang dimaksud.
  - Citra val ikut dilihat tanpa label. Ini dinyatakan di laporan.
- **Encoder SSL baru** (RQ1-08…11, 14, 16) memakai pool yang sama (`--pool h1200`), 100 epoch (seperti E1), dan hiperparameter default dari paper aslinya. Hiperparameter setiap metode dicatat dalam adendum sebelum pre-training dimulai. Tidak ada tuning berdasarkan hasil fine-tuning.

## Arm
| Kode | Inisialisasi / metode | Backbone | Program | Prio |
|---|---|---|---|---|
| RQ0-01 | ImageNet (rujukan, dari antrean RQ0) | ResNet34 | — | — |
| RQ1-01 | Scratch | ResNet34 | flag ada | A |
| RQ1-02 | DINO (E1) | ResNet34 | `--init-from` | A |
| RQ1-03 | DINO + ChronoCon (E1) | ResNet34 | `--init-from` | A |
| RQ1-04 | ImageNet + LLRD | ViT-S | flag ada | A |
| RQ1-05 | MAE (E1) + LLRD | ViT-S | `--init-from` | A |
| RQ1-06 | DINO (E1) + LLRD | ViT-S | `--init-from` | A |
| RQ1-07 | ImageNet → regresi bone age RSNA → fine-tune | ResNet34 | baru | A |
| RQ1-08 | SimCLR | ResNet34 | baru | A |
| RQ1-10 | BYOL | ResNet34 | baru | A |
| RQ1-12 | SupCon + CE | ResNet34 | baru | A |
| RQ1-13 | **Modifikasi:** SupCon berbobot jarak grade | ResNet34 | kecil | A |
| RQ1-14 | SSL normalitas grade 0 (gaya SS-FewSOME), skor kontinu | ResNet34 | baru | A |
| RQ1-15 | **Modifikasi:** skor normalitas sebagai gate 0 vs ≥ 1 | ResNet34 | kecil | A |
| RQ1-17a/b | Linear probe / LP-FT dari ImageNet | ResNet34 | kecil (LP-FT mempertahankan kepala) | A |
| RQ1-17c/d | Linear probe / LP-FT dari SSL terbaik | ResNet34 atau ViT-S | idem | A |
| RQ1-18a/b | SKELEX: linear probe / fine-tune LLRD | ViT-L | baru (loader); **ukur 1 epoch dulu** | A |
| RQ1-09 | MoCo v3 | ResNet34 | baru | B |
| RQ1-11 | iBOT | ViT-S | baru | B |
| RQ1-16 | **Modifikasi:** penyelarasan progresi antarkunjungan (MMD) sebagai ganti ChronoCon | ResNet34 | baru | B |
| RQ1-19 | RAD-DINO / BiomedCLIP di-fine-tune | ViT | baru (loader) | B |

Arm yang memerlukan kode baru baru masuk antrean setelah flag atau modulnya diberi unit test dan lulus smoke test (`--limit`). Rincian implementasi yang tidak memengaruhi kontras (misalnya bobot loss SupCon dan suhu) dikunci dalam adendum bertanggal sebelum run pertama arm itu.

## Pilihan berbasis validasi (dikunci sebelum kontras yang membutuhkannya dihitung)
Semua pilihan di bawah memakai **rata-rata BACC4 validasi** dari run arm tersebut, bukan test. Setiap pilihan dicatat di `STATUS.md` beserta tanggalnya, sebelum kontras yang bergantung padanya dihitung.
- **"Metode terbaik yang sudah ada"** (pembanding kedua untuk modifikasi):
  - Kandidatnya semua arm *metode ada* prioritas A dengan fine-tuning penuh, yaitu RQ0-01, RQ1-01…08, 10, 12, 14, dan 18b.
  - Dipilih per fraksi label: satu untuk 100% dan satu untuk 10%. Pilihan dibuat setelah semua kandidat selesai.
  - Kalau ada kandidat yang tidak selesai, pilihan dibuat dari kandidat yang tersedia, dan hal ini dilaporkan.
- **"SSL terbaik"** untuk RQ1-17c/d: arm SSL di antara RQ1-02, 03, 05, 06, 08, dan 10 dengan rata-rata BACC4 validasi tertinggi dari 6 run (satu encoder untuk kedua fraksi).

## Kontras dan keluarga Holm
Diputuskan pengguna pada 2026-11-11: **satu keluarga Holm per sub-pertanyaan**, bukan satu keluarga untuk seluruh RQ1 (±48 kontras). Alasannya, dengan 3 run per sel, keluarga sebesar itu hampir tidak punya daya uji. Ini menggantikan "Holm di dalam satu RQ" di `docs/KONFIGURASI.md` untuk RQ1.

Setiap kontras dihitung pada 100% dan pada 10% label. Kedua fraksi masuk keluarga yang sama. Ukuran setiap keluarga tetap. Kontras yang arm-nya tidak selesai dilaporkan sebagai "tidak dijalankan", dan p-value-nya diperlakukan sebagai 1 saat koreksi Holm, sehingga keluarga tidak mengecil setelah hasil terlihat.

**F1: pre-training / SSL vs rujukan ImageNet ResNet34 (RQ0-01).** 8 kontras × 2 = 16.
RQ1-01, RQ1-02, RQ1-07, RQ1-08, RQ1-10, RQ1-12, RQ1-14, RQ1-18b, masing-masing − RQ0-01.

**F2: antar-metode dan arsitektur.** 7 kontras × 2 = 14.
- RQ1-03 − RQ1-02 (pengaruh ChronoCon)
- RQ1-04 − RQ0-01 (ViT-S vs ResNet34, keduanya ImageNet-1k). Masih tercampur dengan resep: ViT-S memakai LLRD + warmup, ResNet34 tidak. Kesimpulannya ditulis sebagai "ViT-S dengan resepnya" vs "ResNet34 dengan resepnya".
- RQ1-05 − RQ1-04 (MAE vs ImageNet, ViT-S)
- RQ1-06 − RQ1-04 (DINO vs ImageNet, ViT-S)
- RQ1-08 − RQ1-02 (SimCLR vs DINO)
- RQ1-10 − RQ1-02 (BYOL vs DINO)
- RQ1-18b − metode terbaik yang sudah ada selain SKELEX (dipilih di validasi dengan cara yang sama)

**F3: modifikasi.** 4 kontras × 2 = 8.
- RQ1-13 − RQ1-12 (versi asli)
- RQ1-13 − metode terbaik yang sudah ada
- RQ1-15 − RQ1-14 (versi asli)
- RQ1-15 − metode terbaik yang sudah ada

**F4: strategi fine-tuning.** 5 kontras × 2 = 10.
- RQ1-17a − RQ0-01 (linear probe vs fine-tune penuh, ImageNet)
- RQ1-17b − RQ0-01 (LP-FT vs fine-tune penuh, ImageNet)
- RQ1-17c − arm SSL terbaik (linear probe vs fine-tune penuh)
- RQ1-17d − arm SSL terbaik (LP-FT vs fine-tune penuh)
- RQ1-18a − RQ1-18b (SKELEX linear probe vs fine-tune)

**F5: prioritas B (hanya bila dijalankan, keluarga sendiri, sehingga F1–F4 tidak bergantung pada dijalankan atau tidaknya arm B).** Hingga 7 kontras × 2.
- RQ1-09 − RQ0-01, RQ1-09 − RQ1-02
- RQ1-11 − RQ1-05, RQ1-11 − RQ1-06
- RQ1-16 − RQ1-03 (versi asli), RQ1-16 − metode terbaik yang sudah ada
- RQ1-19 − RQ0-01

Ukuran F5 ditetapkan saat arm B diputuskan untuk dijalankan, sebelum run pertamanya.

**Uji:** BACC4, bootstrap pasien berpasangan, 10.000 replikasi, run ikut diambil ulang (`compare.py`).

**Sekunder (p tidak dikoreksi):**
- selisih QWK dan ICC skor total tangan
- recall per grade
- proporsi prediksi grade 0
- BACC4/QWK validasi dan epoch terpilih
- celah train−val pada checkpoint terpilih (seperti `e8.py gap`)

## Aturan putusan (BACC4)
- **Bermakna:** Holm p < 0,05 dan |selisih| ≥ 0,05.
- **Kecil:** Holm p < 0,05 dan |selisih| < 0,05.
- **Tidak terbukti:** selain itu. Hasil yang tidak signifikan bukan bukti kesetaraan.

Klaim "modifikasi lebih baik" hanya dibuat kalau modifikasi unggul **bermakna atau kecil** terhadap versi asli **dan** terhadap metode terbaik yang sudah ada, pada fraksi yang sama. Klaim "baru" juga memerlukan pencarian literatur terarah (aturan 10 `CLAUDE.md`). Semua hasil dilaporkan, apa pun arahnya.

## Urutan run (bukan bagian dari aturan statistik)
1. RQ1-01…06 (flag ada, ±14 jam kasar), dengan urutan arm berselang seperti RQ0.
2. RQ1-17a/b (ImageNet), setelah flag LP-FT selesai.
3. Arm yang membutuhkan kode baru, sesuai prioritas di `PLAN_RQ.md`. RQ1-07 memerlukan izin unduhan dataset RSNA. RQ1-18 diukur 1 epoch dulu.
4. Pilihan "SSL terbaik", lalu RQ1-17c/d.
5. Pilihan "metode terbaik", lalu kontras F2 (RQ1-18b) dan F3.

## Di luar cakupan
- Kepala selain balanced softmax (RQ2), label noise (RQ5).
- Encoder E1 DINO + ChronoCon ViT-S (ada sebagai artefak, tetapi tidak ada di `PLAN_RQ.md`).
- Pool pre-training lain (RAM-W600), tuning hiperparameter per encoder.

## Adendum 1 (2026-11-11): resep RQ1-17 (linear probe dan LP-FT)
Ditulis dan di-commit sebelum run LP mana pun. Keputusan pengguna pada 2026-11-11. Kontras F4 tidak berubah.

**Kode:**
- Linear probe: `--freeze-backbone` (dengan `--pretrained` atau `--init-from`). Hanya kepala yang dilatih, dan BatchNorm backbone tetap dalam mode eval.
- Tahap 2 LP-FT: `--init-from <run LP>/model_best.pt --keep-head`, yaitu checkpoint LP dimuat lengkap beserta kepalanya.
- Kode: `train.py: check_init_args()`, `models.py: load_full()`. Test: `tests/test_rq1.py`.

**1. Learning rate linear probe, dipilih di validasi.**
- **Grid:** lr ∈ {1e-4, 1e-3, 1e-2}.
- **Setelan:** ImageNet ResNet34, `--pretrained --freeze-backbone`, 10% label d0 s42. Selebihnya sama dengan resep RQ0-01 (balanced softmax, aug `base` di-seed, 50 epoch dan minimal 2.000 step, monitor QWK).
  - Nama run: `rq1-17tune_lr{lr}_resnet34_imagenet_lp_b2_f0.1_d0_s42`.
- **Pemilihan:** lr dengan BACC4 validasi tertinggi. Kalau seri sampai 3 desimal, dipilih lr yang lebih kecil. Test run tuning tidak dibaca untuk pemilihan, dan run tuning tidak masuk kontras mana pun.
- **Pencatatan:** lr terpilih dicatat bertanggal di `STATUS.md` dan di bagian bawah adendum ini sebelum run RQ1-17 pertama.
- **Cakupan:** lr ini dipakai untuk **semua** run linear probe (RQ1-17a, RQ1-17c, dan tahap 1 LP-FT RQ1-17b/d), tanpa LLRD dan tanpa warmup.
  - Kalau "SSL terbaik" ternyata ViT-S, lr ini tetap dipakai walaupun dipilih di ResNet34 ImageNet. Batasan ini dicatat di laporan.

**2. Tahap 2 LP-FT, satu faktor terhadap fine-tune penuh.**
- Resep sama dengan fine-tune penuh untuk backbone yang sama:
  - ResNet34: lr 1e-4, tanpa warmup.
  - ViT-S: LLRD 0,85, lr 1e-4, warmup 10%.
  - Keduanya: 50 epoch dan minimal 2.000 step, balanced softmax, aug `base` di-seed, monitor QWK.
- Satu-satunya perbedaan dari fine-tune penuh adalah kepala yang sudah dilatih LP, bukan kepala acak.
- Tahap 2 selalu memakai run LP dengan desain yang sama (fraksi, draw, dan seed yang sama).
- Run LP tahap 1 untuk RQ1-17b/d **adalah** run RQ1-17a/c. Jadi RQ1-17b/d tidak menambah run LP baru, cukup memakai `model_best.pt` dari RQ1-17a/c.

**3. Biaya (kasar).** RQ1-17a/b (ImageNet): 6 run LP + 6 run FT ≈ 3 jam. RQ1-17c/d setara, tergantung backbone SSL terbaik.

**lr linear probe terpilih:** *(diisi setelah 3 run tuning, sebelum run RQ1-17 pertama)*
