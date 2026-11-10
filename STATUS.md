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
- **Belum ada run sungguhan.**

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

## Langkah berikutnya
1. Minta izin, lalu luncurkan antrean RQ0 (12 run, ±5,5 jam kasar): `setsid nohup bash scripts/run_rq0.sh all >> runs/rq0.log 2>&1 < /dev/null &`.
2. Paralel tanpa GPU: tulis `rq0.py` (analisis + uji reproduksi vs E8), lalu kode RQ1 / RQ2 sesuai prioritas di `docs/PLAN_RQ.md`.
3. Push ke remote kapan pun pengguna memutuskan.

## Pertanyaan terbuka untuk pengguna atau dosen
- Verifikasi manusia untuk SLR-M (`/data/Smart-X/SLR-RA/slr-methods/verification/`) belum dikerjakan.
