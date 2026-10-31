# AMZN RL - Kết quả test, model chính run 09

Kết quả test được tổng hợp từ **`logs/assets_test.csv`** và sử dụng thống nhất trong JSON, README, log và báo cáo PDF. Model chính là **run 09, seed 8, checkpoint 32.768 bước**, chọn theo validation **208,514.04 USD**.

## File nộp bài

- **`output/AMZN_NOP_BAI.zip`**: gói nộp đầy đủ mã nguồn, cấu hình, dữ liệu, 10 cặp model/normalizer, log và báo cáo.
- **`output/pdf/BAO_CAO_AMZN.pdf`**: báo cáo tiếng Việt, bảng 10 run, biểu đồ giá và tài sản, phân tích model run 09.
- **`models/run_09/agent.zip`** và **`models/run_09/normalizer.pkl`**: cặp model chính.
- `logs/assets_test.csv`: log tài sản test; 5.020 dòng, 21 cột, 10 run.
- `logs/assets_train.csv`: log PPO, validation và replay train.
- `logs/result.json`: thông số, kết quả từng run, các đỉnh, tổng hợp và SHA256.
- `logs/run.log`: tiến trình train và tổng hợp kết quả test.
- `logs/checks.log`: kết quả kiểm thử mã nguồn và kiểm tra số liệu.

## Kết quả model run 09

| Chỉ số | Kết quả |
|---|---:|
| Vốn đầu kỳ | 100,000.00 USD |
| Tài sản cuối test | **919,870.90 USD** |
| Lợi nhuận | **819,870.90 USD (+819.87%)** |
| Đỉnh tài sản test | **952,874.71 USD**, ngày **2026-10-28** |
| Hoàn thành mục tiêu 1 triệu USD | 91.9871% theo tài sản cuối kỳ |
| Tài sản cuối theo High phiên trước | 962,709.15 USD |
| Max drawdown | 3.83% |
| CAGR (252 phiên/năm) | 205.32% |
| Sharpe (lãi suất phi rủi ro bằng 0) | 4.8293 |
| Reward tích lũy | 221.808019 |
| Tổng phí | 858,854.73 USD |
| Số phiên | 501 |
| Yêu cầu mua / bán / HOLD | 76 / 79 / 346 |
| Guard can thiệp / số lệnh thực hiện | 3 / 158 |
| Số lần phạt không giao dịch | 0 |

Run 09 có tài sản cuối test và đỉnh test cao nhất trong 10 lượt. Chưa run nào đạt 1.000.000 USD trên test. Giá và tài sản được lấy trực tiếp từ CSV; dòng `step=0` là vốn đầu kỳ và được loại khi đếm phiên, quyết định, giao dịch.

## Kết quả đủ 10 run

| Run | Seed | Validation (USD) | Cuối test (USD) | Lợi nhuận | Đỉnh test (USD) | Số lệnh |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 0 | 142,213.15 | 609,508.59 | +509.51% | 624,809.64 | 133 |
| 02 | 1 | 208,107.72 | 862,022.98 | +762.02% | 892,951.28 | 152 |
| 03 | 2 | 149,129.91 | 722,543.38 | +622.54% | 750,511.03 | 144 |
| 04 | 3 | 173,809.15 | 816,420.52 | +716.42% | 845,712.81 | 143 |
| 05 | 4 | 114,971.84 | 645,160.40 | +545.16% | 668,755.32 | 132 |
| 06 | 5 | 148,998.71 | 842,569.30 | +742.57% | 872,799.63 | 149 |
| 07 | 6 | 132,464.25 | 766,708.75 | +666.71% | 791,197.26 | 146 |
| 08 | 7 | 109,501.49 | 683,039.28 | +583.04% | 708,225.27 | 139 |
| 09 **(chính)** | 8 | 208,514.04 | 919,870.90 | +819.87% | 952,874.71 | 158 |
| 10 | 9 | 183,524.30 | 885,338.12 | +785.34% | 917,102.95 | 157 |

Tài sản cuối test trung bình **775,318.22 USD**, trung vị **791,564.64 USD**, độ lệch chuẩn mẫu **106,738.66 USD**. Khoảng kết quả: **609,508.59 - 919,870.90 USD**. Cả 10 lượt có tài sản cuối cao hơn vốn ban đầu.

## Dữ liệu và mô hình

| Phần | Phạm vi | Vai trò |
|---|---|---|
| Fit | 2014-10-29 đến 2022-10-28 | BC, oracle và PPO |
| Validation | 2022-10-31 đến 2024-10-29 | Chọn checkpoint và model chính |
| Test | 2024-10-30 đến 2026-10-29 | 501 phiên test cho mỗi run |

10 seed 0-9 dùng cùng cấu hình risk_2; mỗi lượt có 32.768 transition PPO, tổng 327.680 transition. Run 09 dùng checkpoint 32.768 bước. Encoder MLP tổng hợp cửa sổ 30 phiên bằng last/mean/std, trung bình 5 phiên gần nhất và chênh đầu/cuối cửa sổ; actor/critic [64, 64]. BC và guidance dùng mẫu pha giữa phân phối tự nhiên và cân bằng HOLD/BUY/SELL. Checkpoint được đánh giá sau cập nhật PPO ở 16.384 và 32.768 bước.

PPO: 8 env, n_steps 512, batch 256, 5 epoch, LR 0,0002 giảm đến 10%, gamma 0,9995, GAE 0,98, clip 0,2, target KL 0,04, vf 0,25. BC: 16 epoch, LR 0,0003, batch 256, 8 trajectory; guidance 0,02 về 0. Entropy risk_2: 0,075 về 0,006.

Đỉnh PPO run 09: **114,408.73 USD**, ngày 2015-11-11. Đỉnh replay fit của run 09: **1,025,351.37 USD**, ngày 2020-07-09; replay dừng sau 1433 phiên theo điều kiện thắng. Replay fit là kết quả trong mẫu, được trình bày riêng với kết quả test.

## Luật và cách đọc CSV

- Vốn 100.000 USD; thắng khi tài sản đạt 1.000.000 USD.
- Mua 49%, 58%, 66%, 75%, 83% tiền mặt; bán 21%, 33%, 45%, 57%, 69% cổ phiếu; phí 4% mỗi chiều.
- Sau 19 phiên không giao dịch, phạt 3.751 USD; thua khi tài sản dưới 1.000 USD hoặc tiền mặt không lớn hơn -5.000 USD.
- Luật môi trường: mua Low, bán High, một hành động/phiên, cổ phiếu lẻ; tài sản = cash + shares × Close. Cột `price` lưu giá khớp của từng lệnh.
- Reward: `100*ln(V/V_truoc) - 0.1*max(0, DD-DD_truoc) - invalid + thuong_ket_thuc`; thắng +200, thua -100.
- Action 0: HOLD; 1-5: mua; 16-20: bán. `decision` là yêu cầu, `action` là lệnh sau Guard.

| Cột / nhóm | Ý nghĩa |
|---|---|
| `run, stage` | Run 1-10; optimization / validation / train_replay / test |
| `rank, episode, step` | Môi trường, episode, bước; step=0 là vốn đầu kỳ |
| `date, price, close, price_change_pct` | Ngày, giá khớp, Close và biến động giá |
| `decision, action, guard` | Quyết định, lệnh thực hiện, Guard |
| `cash, shares, assets, asset_change` | Tiền mặt, cổ phiếu, tài sản, thay đổi tài sản |
| `fee, penalty` | Phí giao dịch và tiền phạt |
| `reward, cumulative_reward, is_peak` | Reward, tích lũy, đỉnh mới |

Khi đọc train, tách theo `run, stage, rank, episode`; mỗi episode có vốn đầu kỳ riêng. CSV không lưu định dạng in đậm; các đỉnh được in đậm trong PDF.

## Mã nguồn và chạy lại

Sáu module: `agent.py` (PPO/BC/policy), `environment.py` (luật và sổ tài sản), `data.py` (dữ liệu/metric), `baseline.py` (chiến lược tham chiếu), `main.py` (train/đóng gói), `report.py` (đồng bộ kết quả và tạo PDF). Kiểm thử tập trung tại `tests/test_core.py`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

Đồng bộ JSON, README và log từ CSV hiện tại, sau đó dựng PDF và ZIP:

```powershell
.\.venv\Scripts\python.exe -B report.py --sync
.\.venv\Scripts\python.exe -B main.py --package
```

Chỉ đồng bộ JSON, README, log và giữ nguyên PDF hiện có: `python -B report.py --sync-only`.

Chỉ dựng lại PDF từ kết quả đã đồng bộ: `python -B report.py`. Train chiến dịch mới trong thư mục riêng: `python -B main.py --work _run/chien_dich_moi --run-only`. `main.py --reproduce` dùng cho chế độ tái lập model; cập nhật tài liệu từ kết quả CSV bằng `report.py --sync`.

ZIP không chứa môi trường ảo, cache hoặc thư mục tạm. Nguồn và bản quyền thư viện PPO: `THIRD_PARTY_NOTICES.md`.
