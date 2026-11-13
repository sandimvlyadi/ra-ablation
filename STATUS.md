# STATUS

**Terakhir diperbarui:** 2026-11-13, oleh Claude (sesi di `ra-ablation`).

## Posisi sekarang
- Repo disiapkan **secara lokal, belum di-commit dan belum di-push**. Pengguna memutuskan kapan commit pertama dan push dilakukan.
- Isi repo:
  - kode `ramscore` (dari `slr-ra-experiments` @ b6e0135; modul prototipe dibuang)
  - 5 encoder SSL E1 dengan sha256
  - ROI RheumaVIT dengan sha256
  - symlink data
  - `CLAUDE.md`, dan `docs/` (RUNBOOK, PLAN_RQ, KONFIGURASI, HASIL_LAMA, sources)
- `uv sync` dan pytest: 209 passed (2026-11-11).
- **2026-11-11 (sesi ini):**
  - `rules/rq0_prespecification.md` ditulis dan dikunci (bobot ConvNeXt `fb_in1k`), lalu masuk commit awal lokal.
  - `scripts/run_rq0.sh` ditulis. Smoke test di scratchpad lulus: 12/12 run dengan `metrics.json`.
  - Ukuran 1 epoch (100% label): ResNet34 54 s, ConvNeXt-Tiny 91 s (±1,9×, VRAM 4,3 GB). Perkiraan RQ0 **±5,5 jam (kasar)**, naik dari ±4 jam di `PLAN_RQ.md`.
  - Catatan: setelah 1 epoch, ConvNeXt-Tiny (in12k) masih memprediksi semua sendi sebagai grade 0 (val QWK 0). Satu epoch belum informatif, tetapi resep tanpa warmup mungkin kurang cocok untuk ConvNeXt. Sesuai draf aturan, resep tidak disetel ulang.
  - Commit lokal: `63df280` (repo awal) dan `b877559` (aturan + antrean RQ0). Belum di-push.
  - **Antrean RQ0 diluncurkan 2026-11-11 00:54** dengan izin pengguna (log `runs/rq0.log`, berhenti dengan `kill -TERM -- -$(cat runs/rq0/queue.pid)`). Kalau PC mati, jalankan ulang perintah yang sama; run yang sudah selesai dilewati.
  - `code/src/ramscore/rq0.py` + `tests/test_rq0.py` ditulis (pytest 219 passed), di-commit lokal `b9e0dd6`.
  - `rules/rq1_prespecification.md` disetujui pengguna dan di-commit lokal (`4fccaab`) sebelum run RQ1 mana pun.
  - `scripts/run_rq1.sh` (tahap 1, RQ1-01…06, 36 run, ±13 jam kasar) ditulis. Smoke test lulus 36/36, dan encoder E1 termuat dengan benar.

## Keputusan pengguna
| Tanggal | Keputusan |
|---|---|
| 2026-11-10 | Ablasi per RQ di folder/repo baru yang bersih; arm lama dilatih ulang di sini (tidak diimpor) |
| 2026-11-10 | Cakupan: perbandingan sistematis + modifikasi, untuk semua RQ (bukan hanya RQ1), ditambah RQ6 kombinasi; lihat `docs/PLAN_RQ.md` |
| 2026-11-11 | Dokumentasi setiap langkah: perintah → kode → output (`docs/RUNBOOK.md`) |
| 2026-11-11 | Eksekusi dimulai pengguna di sesi Claude Code baru di folder ini |
| 2026-11-11 | RQ0-02 memakai `convnext_tiny.fb_in1k` (ImageNet-1k, sama dengan ResNet34); diunduh ke cache HF |
| 2026-11-11 | Ukuran sampel tetap 3 seed + 3 draw per arm (6 run), untuk semua RQ |
| 2026-11-11 | Monitor checkpoint tetap QWK validasi, untuk semua RQ |
| 2026-11-11 | Commit awal hanya lokal (tanpa push) |
| 2026-11-11 | Holm RQ1 per sub-pertanyaan (F1 vs ImageNet, F2 antar-metode, F3 modifikasi, F4 strategi fine-tuning, F5 prio B), bukan satu keluarga ±48 kontras; lihat `rules/rq1_prespecification.md` |
| 2026-11-11 | ImageNet ViT-S (RQ1-04, LP/LP-FT ViT) memakai `vit_small_patch16_224.augreg_in1k`, bukan in21k_ft_in1k (E2/E6), supaya baseline ImageNet-1k setara dengan ResNet34/ConvNeXt; diunduh ke cache HF |
| 2026-11-11 | RQ1-17: lr linear probe dipilih di validasi dari {1e-4, 1e-3, 1e-2} (10% d0 s42, ImageNet ResNet34); tahap 2 LP-FT memakai resep FT penuh (lr 1e-4 / LLRD ViT) |
| 2026-11-11 | Hasil tuning adendum 1 (validasi saja): lr linear probe = **1e-2** (val BACC4 0,382 vs 0,364 @1e-3 dan 0,330 @1e-4); tepi atas grid, grid tidak diperluas |
| 2026-11-11 | RQ0 lolos uji reproduksi; antrean RQ1 utama (RQ1-01..06) diluncurkan |
| 2026-11-13 | Hasil tuning adendum 3 (validasi saja): **λ SupCon = 1,0** (val BACC4 0,387 vs 0,379 @0,5 dan 0,377 @0,1); tepi atas grid, grid tidak diperluas. Unduh SKELEX (HF `skhoha/SKELEX`, CC-BY-NC-ND-4.0) diizinkan dan selesai; commit + push adendum 5 + 6 diizinkan setelah smoke bone age |
| 2026-11-13 | Insiden: `run_rq1.sh` diedit saat `tune12` berjalan; bash membaca skrip bertahap sehingga keluar dengan syntax error **setelah** 3 run tuning selesai (tidak ada run lain terpengaruh). Perbaikan: antrean kini menjalankan salinan beku di `runs/.queue_snapshots/` |
| 2026-11-13 | RQ1-14 = M197 tahap 1 (SS-FewSOME pada sendi grade 0 train berlabel), grade dari skor saja dengan cut-point disetel di validasi; RQ1-15 = skor sebagai gate 0 vs ≥1 di atas classifier rujukan. Adendum 4 (SimCLR/BYOL) di-commit + push. Unduh RSNA Bone Age (Kaggle) diizinkan |
| 2026-11-13 | "ya boleh luncurkan berkelanjutan, prio A dulu": Claude boleh meluncurkan setiap tahap RQ1 prio A sendiri setelah adendumnya di-commit dan smoke test lulus, dengan kabar setiap peluncuran. Commit/push dan unduhan (RSNA, SKELEX) belum diizinkan secara umum, jadi tetap ditanyakan. Prio B (09, 11, 16, 19) diputuskan nanti |
| 2026-11-13 | RQ1-13 sudah dipublikasikan oleh M219 (negatif berbobot jarak grade, domain adaptation KL lutut): dilaporkan sebagai **adaptasi M219, tanpa klaim baru**. λ SupCon dipilih di validasi dari {0,1; 0,5; 1,0} (adendum 3) |
| 2026-11-13 | Urutan kerja: **RQ1 diselesaikan dulu**; tahap 1 RQ2 tidak diluncurkan paralel (GPU boleh menganggur selama kode RQ1 ditulis). Draf aturan RQ2 tetap belum di-commit |
| 2026-11-13 | Kriteria "metode terbaik" RQ1: **opsi (a) + analisis sensitivitas**. Kriteria utama tetap val BACC4; ditambah pembanding sensitivitas "terbaik dengan val QWK ≥ rujukan ImageNet ResNet34" (p mentah, tanpa Holm). Adendum 2 `rules/rq1_prespecification.md`; aturan yang sama ditulis di draf RQ2 |
| 2026-11-12 | Modul analisis `rq1.py` ditulis; laporan RQ1 sementara (`reports/rq1.md`). Checkpoint smoke test sesi lama di `/tmp` (12 GB, kuota tmpfs penuh) dihapus, log/metrics smoke tetap |

## Langkah berikutnya
1. ~~Antrean RQ0~~ selesai 2026-11-11 08:33 (12/12 run).
2. ~~Analisis RQ0~~ selesai 2026-11-11: uji reproduksi identik dengan E8 `base` (prediksi identik 1,000 pada 6/6 run, selisih BACC4 0,000), jadi RQ1 tidak ditahan. ConvNeXt-Tiny − ResNet34: BACC4 +0,020 (100%) dan +0,018 (10%), Holm p = 0,392 untuk keduanya → tidak terbukti. Laporan: `reports/rq0.md`.
3. **RQ1 `all` diluncurkan 2026-11-11 09:33 dengan izin, selesai 20:37** (36 run, ±11 jam; total RQ1 51 run dengan `metrics.json`). Analisis sementara (RQ1-01…06, 17a/b) 2026-11-12: `reports/rq1.md`. Semua 16 kontras yang bisa dihitung "tidak terbukti" (Holm p = 1,000). Pilihan "metode terbaik" dan "SSL terbaik" masih sementara karena kandidat RQ1-07/08/10/12/14/18b belum dijalankan. Cek: `bash scripts/run_rq1.sh`. Perintah resume: `setsid nohup bash scripts/run_rq1.sh all >> runs/rq1.log 2>&1 < /dev/null &` (±13 jam kasar).
4. RQ1-17: tuning lr selesai (lr probe = 1e-2, `1b54a0e`). **`lp17` diluncurkan 2026-11-11 02:40 dengan izin, selesai 07:30 (12/12 run + 3 tuning).** Perintah resume: `LP_LR=1e-2 setsid nohup bash scripts/run_rq1.sh lp17 >> runs/rq1.log 2>&1 < /dev/null &` (12 run, ±3,8 jam kasar).
4b. **Rangkaian RQ1 prio A diluncurkan 2026-11-13 01:10** (izin berkelanjutan): `supcon12` (λ 1,0) → `norm1415` → `boneage07` → `ssl0810`, ±22 jam kasar. Log `runs/rq1.log`; hentikan dengan `kill -TERM -- -$(cat runs/rq1/queue.pid)`. Kalau PC mati, jalankan ulang: `setsid nohup bash -c 'cd /data/Smart-X/ra-ablation; SUPCON_W=1.0 bash scripts/run_rq1.sh supcon12; bash scripts/run_rq1.sh norm1415; bash scripts/run_rq1.sh boneage07; bash scripts/run_rq1.sh ssl0810' >> runs/rq1.log 2>&1 < /dev/null &` (run selesai dilewati, pre-training di-resume).
5. **RQ1-12/13 (SupCon):** adendum 3 di-commit `829e3f2`; `tune12` selesai 2026-11-13 00:59, **λ = 1,0** (validasi). Berikutnya `SUPCON_W=1.0 … supcon12` (12 run, ±5,4 jam kasar).
6. **RQ1-08/10 (SimCLR, BYOL):** adendum 4 di-commit `4bcf033`; 1 epoch SimCLR 145 s (±4 jam per pre-training). `ssl0810` ±12–13 jam kasar.
7. **RQ1-14/15 (normalitas M197 + gate):** kode + 8 test, smoke 12/12 (2026-11-13), adendum 5.
8. **RQ1-07 (bone age):** RSNA disiapkan (12.611 citra 512 px, zip dihapus), kode + 3 test, smoke 6/6, adendum 6; 1 epoch 91 s.
9. **RQ1-18 (SKELEX):** dikonversi ke timm dan diverifikasi (cosine 1,000000), kode + 3 test, smoke OK, adendum 7 ditulis (belum di-commit). Rangkaian prio A (supcon12, norm1415, boneage07, ssl0810) selesai 2026-11-13 21:26, 96 run dengan metrics.json, tanpa gagal. 1 epoch SKELEX diukur (adendum 7): 18b 403 detik, 18a 97 detik train per epoch 100%, total `skelex18` ±30 jam (kasar); 2026-11-13 pengguna memilih "17c/d dulu, lalu SKELEX penuh". SSL terbaik (validasi, 6 run) = RQ1-02 DINO ResNet34 (BACC4 val 0,451; 05 0,440; 08 0,436; 10 0,430; 03 0,426; 06 0,422). Tahap baru `lp17ssl` (smoke 12/12; refaktor `lp_pair` tidak mengubah argumen 17a/b). Rangkaian `lp17ssl` → `skelex18` diluncurkan 2026-11-13 22:17; atas permintaan pengguna (PC dimatikan setelah 17c/d) pembungkus rangkaian dihentikan 22:44 sehingga hanya `lp17ssl` yang berjalan. **Sesi berikutnya:** cek 12 run 17c/d punya metrics.json, lalu `setsid nohup bash scripts/run_rq1.sh skelex18 >> runs/rq1.log 2>&1 < /dev/null &` (±30 jam, kasar).
10. Sisa RQ1 (prio A): lalu RQ1-17c/d setelah "SSL terbaik" final, `rq1 gap`, analisis akhir.
11. Push ke remote kapan pun pengguna memutuskan.

## Pertanyaan terbuka untuk pengguna atau dosen
- ~~(2026-11-12) Kriteria "metode terbaik"~~ diputuskan 2026-11-13: opsi (a) + sensitivitas (adendum 2 RQ1, belum di-commit). Pilihan sensitivitas sementara: 100% DINO ResNet34, 10% ImageNet ResNet34 (rujukan); laporan `reports/rq1.md` diperbarui (kontras utama tidak berubah).
- `rq1 gap` (GPU, ±20–30 menit kasar) belum dijalankan.
- Draf `rules/rq2_prespecification.md` (sudah memuat sensitivitas) menunggu tinjauan pengguna sebelum di-commit.
- Verifikasi manusia untuk SLR-M (`/data/Smart-X/SLR-RA/slr-methods/verification/`) belum dikerjakan.
