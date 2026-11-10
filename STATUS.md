# STATUS

**Terakhir diperbarui:** 2026-11-11, oleh Claude (sesi di `ra-ablation`).

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

## Langkah berikutnya
1. Tunggu antrean RQ0 selesai (±5,5 jam kasar). Cek dengan `bash scripts/run_rq0.sh`.
2. Jalankan `cd code && uv run --group analysis python -m ramscore.rq0`, periksa uji reproduksi (kalau ada run yang ditandai, RQ1 ditahan), lalu salin laporannya ke `reports/rq0.md`.
3. Setelah RQ0 selesai dan dengan izin: luncurkan `setsid nohup bash scripts/run_rq1.sh all >> runs/rq1.log 2>&1 < /dev/null &` (±13 jam kasar).
4. RQ1-17: flag LP-FT (`93a1918`) dan adendum 1 (`e05125e`; lr probe dipilih di validasi, tahap 2 LP-FT memakai resep FT penuh) sudah di-commit. `run_rq1.sh` punya tahap `tune17` dan `lp17` (smoke lulus). Berikutnya: dengan izin, jalankan `tune17` (3 run, ±20 menit), pilih lr dari BACC4 validasi, catat, lalu `LP_LR=<lr> ... lp17` (±3,8 jam).
5. Paralel tanpa GPU: kode SSL baru / RQ2 sesuai prioritas.
6. Push ke remote kapan pun pengguna memutuskan.

## Pertanyaan terbuka untuk pengguna atau dosen
- Verifikasi manusia untuk SLR-M (`/data/Smart-X/SLR-RA/slr-methods/verification/`) belum dikerjakan.
