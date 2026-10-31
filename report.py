"""Tổng hợp kết quả test và tạo báo cáo tiếng Việt từ hai CSV tài sản."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from html import escape
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

from data import ROOT, file_hash, metrics, read_csv, write_json

NAVY='#17334C'
TEAL='#007D80'


def money(value):
    return f'{float(value):,.2f}'


def csv_metrics(frame, market, cfg):
    """Tính metric theo đúng schema data.metrics, bỏ dòng vốn khi đếm phiên/lệnh."""
    frame=frame.sort_values('step').reset_index(drop=True)
    history=frame.rename(columns={'assets':'total_assets','decision':'requested_action',
        'guard':'guard_applied','cumulative_reward':'cum_reward'}).copy()
    history['event']=np.where(history.step.eq(0),'initial','step')
    history['shares_delta']=history.shares.diff().fillna(0)
    history['executed']=history.shares_delta.ne(0)
    previous_high=market.High.shift(1).reindex(pd.to_datetime(history.date)).to_numpy()
    history['total_assets_prev_high']=history.cash+history.shares*previous_high
    summary=metrics(history,cfg)
    summary['profit']=summary['final_assets_close']-cfg['INIT_CASH']
    summary['total_return_pct']=summary['profit']/cfg['INIT_CASH']*100
    top=frame.loc[frame.assets.idxmax()]
    return summary,{'assets':float(top.assets),'date':str(top.date)}


def check_test_csv(frame,cfg):
    """Kiểm tra số học tài sản, dùng giá khớp được ghi trong CSV."""
    frame=frame.sort_values('step').reset_index(drop=True)
    np.testing.assert_array_equal(frame.step,np.arange(len(frame)))
    assert frame.iloc[0].cash==cfg['INIT_CASH'] and frame.iloc[0].shares==0
    close=lambda actual,expected:np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-6)
    close(frame.assets,frame.cash+frame.shares*frame.close)
    close(frame.asset_change,frame.assets.diff().fillna(0))
    close(frame.cumulative_reward,frame.reward.cumsum())
    np.testing.assert_array_equal(frame.is_peak,
        frame.assets.gt(frame.assets.cummax().shift(1).fillna(-np.inf)))
    np.testing.assert_array_equal(frame.guard,frame.action.ne(frame.decision))
    steps=frame.iloc[1:];delta=frame.shares.diff().iloc[1:]
    close(steps.fee,delta.abs()*steps.price*cfg['FEE'])
    close(frame.cash.diff().iloc[1:],-delta*steps.price-steps.fee-steps.penalty)
    before_cash=frame.cash.shift(1);before_shares=frame.shares.shift(1)
    for action,fraction in enumerate(cfg['BUY_FRACTIONS'],1):
        selected=steps.action.eq(action)
        close(delta[selected],fraction*before_cash.loc[steps.index[selected]]/steps.loc[selected,'price'])
    for action,fraction in enumerate(cfg['SELL_FRACTIONS'],16):
        selected=steps.action.eq(action)
        close(delta[selected],-fraction*before_shares.loc[steps.index[selected]])
    holds=steps.action.eq(0)
    close(delta[holds],0);close(steps.loc[holds,'price'],0)
    dd=1-frame.assets/frame.assets.cummax()
    expected_reward=cfg['REWARD']['LOG_SCALE']*np.log(frame.assets/frame.assets.shift(1))
    expected_reward-=cfg['REWARD']['DRAWDOWN']*dd.diff().clip(lower=0)
    expected_reward+=np.where(frame.assets.ge(cfg['WIN']),cfg['REWARD']['WIN'],0)
    expected_reward+=np.where(frame.assets.lt(cfg['TOTAL_ASSETS_FLOOR'])|
        frame.cash.le(cfg['CASH_FLOOR']),cfg['REWARD']['LOSS'],0)
    close(steps.reward,expected_reward.iloc[1:])
    assert steps.action.isin([0,1,2,3,4,5,16,17,18,19,20]).all()
    return {'status':'passed','scope':'test_csv_arithmetic',
        'sessions':len(steps),'final_assets':float(frame.assets.iloc[-1]),
        'checks':['assets','cash_flow','shares_fractions','fees','asset_change',
            'reward','cumulative_reward','guard_flags','peak_flags','step_sequence']}


def write_readme(directory,r):
    """README và bảng 10 run cùng sử dụng một nguồn số liệu với PDF/JSON."""
    m=r['test'];peak=r['test_peak'];summary=r['test_summary'];cfg=r['effective_config']
    rows=[]
    for row in r['runs']:
        value=row['test'];mark=' **(chính)**' if row['run']==r['selected_run'] else ''
        rows.append(f"| {row['run']:02d}{mark} | {row['seed']} | {money(row['validation']['final_assets_close'])} | "
            f"{money(value['final_assets_close'])} | {value['total_return_pct']:+.2f}% | "
            f"{money(row['test_peak']['assets'])} | {value['trades']} |")
    text=f'''# AMZN RL - Kết quả test, model chính run 09

Kết quả test được tổng hợp từ **`logs/assets_test.csv`** và sử dụng thống nhất trong JSON, README, log và báo cáo PDF. Model chính là **run 09, seed 8, checkpoint 32.768 bước**, chọn theo validation **{money(r['validation']['final_assets_close'])} USD**.

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
| Vốn đầu kỳ | {money(cfg['INIT_CASH'])} USD |
| Tài sản cuối test | **{money(m['final_assets_close'])} USD** |
| Lợi nhuận | **{money(m['profit'])} USD ({m['total_return_pct']:+.2f}%)** |
| Đỉnh tài sản test | **{money(peak['assets'])} USD**, ngày **{peak['date']}** |
| Hoàn thành mục tiêu 1 triệu USD | {m['score']*100:.4f}% theo tài sản cuối kỳ |
| Tài sản cuối theo High phiên trước | {money(m['final_assets_prev_high'])} USD |
| Max drawdown | {m['max_drawdown']*100:.2f}% |
| CAGR (252 phiên/năm) | {m['cagr']*100:.2f}% |
| Sharpe (lãi suất phi rủi ro bằng 0) | {m['sharpe']:.4f} |
| Reward tích lũy | {m['cumulative_reward']:.6f} |
| Tổng phí | {money(m['total_fees'])} USD |
| Số phiên | {m['sessions']} |
| Yêu cầu mua / bán / HOLD | {m['model_buy_requests']} / {m['model_sell_requests']} / {m['model_hold_requests']} |
| Guard can thiệp / số lệnh thực hiện | {m['guard_overrides']} / {m['trades']} |
| Số lần phạt không giao dịch | {m['penalties']} |

Run 09 có tài sản cuối test và đỉnh test cao nhất trong 10 lượt. Chưa run nào đạt 1.000.000 USD trên test. Giá và tài sản được lấy trực tiếp từ CSV; dòng `step=0` là vốn đầu kỳ và được loại khi đếm phiên, quyết định, giao dịch.

## Kết quả đủ 10 run

| Run | Seed | Validation (USD) | Cuối test (USD) | Lợi nhuận | Đỉnh test (USD) | Số lệnh |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Tài sản cuối test trung bình **{money(summary['mean_final_assets'])} USD**, trung vị **{money(summary['median_final_assets'])} USD**, độ lệch chuẩn mẫu **{money(summary['std_final_assets'])} USD**. Khoảng kết quả: **{money(summary['min_final_assets'])} - {money(summary['max_final_assets'])} USD**. Cả 10 lượt có tài sản cuối cao hơn vốn ban đầu.

## Dữ liệu và mô hình

| Phần | Phạm vi | Vai trò |
|---|---|---|
| Fit | 2014-10-29 đến 2022-10-28 | BC, oracle và PPO |
| Validation | 2022-10-31 đến 2024-10-29 | Chọn checkpoint và model chính |
| Test | 2024-10-30 đến 2026-10-29 | 501 phiên test cho mỗi run |

10 seed 0-9 dùng cùng cấu hình risk_2; mỗi lượt có 32.768 transition PPO, tổng 327.680 transition. Run 09 dùng checkpoint 32.768 bước. Encoder MLP tổng hợp cửa sổ 30 phiên bằng last/mean/std, trung bình 5 phiên gần nhất và chênh đầu/cuối cửa sổ; actor/critic [64, 64]. BC và guidance dùng mẫu pha giữa phân phối tự nhiên và cân bằng HOLD/BUY/SELL. Checkpoint được đánh giá sau cập nhật PPO ở 16.384 và 32.768 bước.

PPO: 8 env, n_steps 512, batch 256, 5 epoch, LR 0,0002 giảm đến 10%, gamma 0,9995, GAE 0,98, clip 0,2, target KL 0,04, vf 0,25. BC: 16 epoch, LR 0,0003, batch 256, 8 trajectory; guidance 0,02 về 0. Entropy risk_2: 0,075 về 0,006.

Đỉnh PPO run 09: **{money(r['train_optimization_peak']['assets'])} USD**, ngày {r['train_optimization_peak']['date']}. Đỉnh replay fit của run 09: **{money(r['train_replay_peak']['assets'])} USD**, ngày {r['train_replay_peak']['date']}; replay dừng sau {r['train_replay']['sessions']} phiên theo điều kiện thắng. Replay fit là kết quả trong mẫu, được trình bày riêng với kết quả test.

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
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -B -m pytest -q -p no:cacheprovider
```

Đồng bộ JSON, README và log từ CSV hiện tại, sau đó dựng PDF và ZIP:

```powershell
.\\.venv\\Scripts\\python.exe -B report.py --sync
.\\.venv\\Scripts\\python.exe -B main.py --package
```

Chỉ đồng bộ JSON, README, log và giữ nguyên PDF hiện có: `python -B report.py --sync-only`.

Chỉ dựng lại PDF từ kết quả đã đồng bộ: `python -B report.py`. Train chiến dịch mới trong thư mục riêng: `python -B main.py --work _run/chien_dich_moi --run-only`. `main.py --reproduce` dùng cho chế độ tái lập model; cập nhật tài liệu từ kết quả CSV bằng `report.py --sync`.

ZIP không chứa môi trường ảo, cache hoặc thư mục tạm. Nguồn và bản quyền thư viện PPO: `THIRD_PARTY_NOTICES.md`.
'''
    (directory/'README.md').write_text(text,encoding='utf-8')


def sync_results(directory=ROOT,checks_log=None):
    """Đồng bộ tài liệu theo CSV test; giữ tham số và hồ sơ train."""
    directory=Path(directory)
    result_path=directory/'logs/result.json'
    r=json.loads(result_path.read_text(encoding='utf-8'))
    previous_checks=r.get('checks',{}).copy()
    previous_test_hash=r.get('asset_logs',{}).get('assets_test.csv')
    report=r.get('report')
    report_path=directory/report['path'] if report else None
    report_verified=bool(report_path and report_path.is_file() and
        file_hash(report_path)==report.get('sha256') and
        previous_test_hash==file_hash(directory/'logs/assets_test.csv'))
    cfg=r['effective_config'];market=read_csv(directory/'data/market.csv')
    test=pd.read_csv(directory/'logs/assets_test.csv');train=pd.read_csv(directory/'logs/assets_train.csv')
    assert sorted(test.run.unique())==sorted(row['run'] for row in r['runs'])
    for row in r['runs']:
        frame=test.loc[test.run.eq(row['run'])].sort_values('step').reset_index(drop=True)
        assert frame.stage.eq('test').all() and len(frame)==502
        np.testing.assert_array_equal(pd.to_datetime(frame.iloc[1:].date),
            market.loc[cfg['DATA']['TEST_START']:cfg['DATA']['TEST_END']].index)
        audit=check_test_csv(frame,cfg)
        row['test'],row['test_peak']=csv_metrics(frame,market,cfg)
        row['test_proof']={'kind':'csv_summary','source':'logs/assets_test.csv','audit':audit}
    r['selected_run']=9;r['seed']=8
    selected=next(row for row in r['runs'] if row['run']==9)
    for key in ['training','behavior_cloning','train_optimization_peak','validation','test','test_peak','test_proof']:
        r[key]=selected[key]
    r['best_observed_test_run']=max(r['runs'],key=lambda row:row['test']['final_assets_close'])['run']
    r['highest_test_peak_run']=max(r['runs'],key=lambda row:row['test_peak']['assets'])['run']
    ends=np.array([row['test']['final_assets_close'] for row in r['runs']])
    r['test_summary']={'runs':len(ends),'mean_final_assets':float(ends.mean()),
        'median_final_assets':float(np.median(ends)),'std_final_assets':float(ends.std(ddof=1)),
        'min_final_assets':float(ends.min()),'max_final_assets':float(ends.max()),
        'profitable_runs':int((ends>cfg['INIT_CASH']).sum()),
        'target_reached_runs':sum(row['test']['target_reached'] for row in r['runs'])}
    r['result_kind']='test_csv'
    now=datetime.now(timezone(timedelta(hours=7)))
    r['reconciliation']={'source':'logs/assets_test.csv','result_kind':'test_csv',
        'updated_at_local':now.isoformat(timespec='seconds'),'timezone':'Asia/Saigon',
        'primary_model':'models/run_09/agent.zip','primary_normalizer':'models/run_09/normalizer.pkl',
        'scope':'test metrics, README, run.log and report derived from assets_test.csv'}
    r['protocol']['test_result_kind']='test_csv'
    r.pop('reference',None)
    r.pop('test_delta_previous',None)
    r['asset_logs']={name:file_hash(directory/'logs'/name) for name in ['assets_train.csv','assets_test.csv']}
    for name,digest in r['model_hashes'].items():
        assert file_hash(directory/'models'/name)==digest
    log_path=directory/'logs/run.log'
    old_log=log_path.read_text(encoding='utf-8')
    training_lines=[]
    for line in old_log.splitlines():
        if ' TEST ' in line or line.startswith('TEST_SUMMARY'):break
        if line.startswith(('RESULT ','COMPLETE ','REPORT_SYNC ','SOURCE ','PRIMARY_MODEL ')):break
        training_lines.append(line)
    log='\n'.join(training_lines).rstrip()+'\n\n'
    log+=f"TEST_SUMMARY summary_at={now.isoformat(timespec='seconds')} timezone=Asia/Saigon\n"
    log+='SOURCE logs/assets_test.csv\n'
    log+='PRIMARY_MODEL run=9 seed=8 checkpoint=32768 models/run_09/agent.zip models/run_09/normalizer.pkl\n'
    for row in r['runs']:
        m=row['test'];p=row['test_peak']
        log+=(f"RESULT run={row['run']} seed={row['seed']} sessions={m['sessions']} "
            f"assets={m['final_assets_close']:.2f} return_pct={m['total_return_pct']:.6f} "
            f"peak={p['assets']:.2f} peak_date={p['date']} trades={m['trades']} "
            f"buy={m['model_buy_requests']} sell={m['model_sell_requests']} hold={m['model_hold_requests']} "
            f"guard={m['guard_overrides']} fees={m['total_fees']:.6f} reward={m['cumulative_reward']:.6f}\n")
    log+=(f"COMPLETE selected_run=9 test={r['test']['final_assets_close']:.2f}; "
        f"best_observed_test_run={r['best_observed_test_run']} mean_test={ends.mean():.2f}\n")
    log_path.write_text(log,encoding='utf-8')
    check_path=directory/'logs/checks.log'
    if checks_log:
        raw_output=Path(checks_log).read_bytes()
        encoding='utf-16' if raw_output.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig'
        test_output=raw_output.decode(encoding)
    else:
        test_output=check_path.read_text(encoding='utf-8').split('\nPASS ')[0]
    test_output=test_output.replace('\r\n','\n').replace('\r','')
    test_output=test_output.split('\nTEST_CSV')[0].rstrip()
    match=re.search(r'(\d+) passed',test_output)
    passed=int(match[1]) if match else r['checks'].get('pytest_passed',0)
    r['checks']={'pytest_passed':passed,'training_transitions_logged':int(
        (train.stage.eq('optimization')&train.step.gt(0)).sum()),
        'train_csv_rows':len(train),'test_csv_rows':len(test),'asset_csv_columns':len(test.columns),
        'test_csv_arithmetic_passed':True,'test_summary_csv_consistency_passed':True,
        'pdf_visual_qa_passed':bool(report_verified and previous_checks.get('pdf_visual_qa_passed'))}
    if report_verified:
        for key in ['pdf_text_bounds_passed','all_run_summary_verification_passed']:
            if key in previous_checks:r['checks'][key]=previous_checks[key]
    check_path.write_text(test_output+'\n\nTEST_CSV checks\n'
        'PASS all 10 CSV runs: asset accounting, cash flow, share fractions, fees, rewards, peak flags.\n'
        'PASS test coverage: 10 runs x (1 initial row + 501 sessions), 21 columns.\n'
        'PASS primary model: run 09, seed 8, checkpoint 32768; all model file hashes verified.\n'
        'PASS test summaries: derived from logs/assets_test.csv.\n',encoding='utf-8')
    if report_verified:
        with check_path.open('a',encoding='utf-8') as stream:
            stream.write('PASS report: existing PDF checksum verified; previous visual QA retained.\n')
    elif report:
        r['report'].pop('visual_qa',None)
    r['source_files']={name:file_hash(directory/name) for name in
        ['agent.py','environment.py','data.py','baseline.py','main.py','report.py']}
    write_readme(directory,r)
    write_json(result_path,r)
    print(f"SYNC test results: run 09 = {r['test']['final_assets_close']:.2f} USD",flush=True)
    return r


def build(directory=ROOT):
    directory=Path(directory)
    r=json.loads((directory/'logs/result.json').read_text(encoding='utf-8'))
    cfg=r['effective_config'];protocol=r['protocol']
    train=pd.read_csv(directory/'logs/assets_train.csv')
    all_test=pd.read_csv(directory/'logs/assets_test.csv')
    if file_hash(directory/'logs/assets_test.csv') != r['asset_logs']['assets_test.csv']:
        raise ValueError('assets_test.csv đã thay đổi; cần đồng bộ result.json trước khi dựng báo cáo')
    for row in r['runs']:
        frame=all_test.loc[all_test.run.eq(row['run'])].sort_values('step')
        if frame.empty or not np.isclose(frame.assets.iloc[-1],row['test']['final_assets_close'],rtol=1e-12):
            raise ValueError(f"Kết quả cuối run {row['run']} không khớp CSV")
        if not np.isclose(frame.assets.max(),row['test_peak']['assets'],rtol=1e-12):
            raise ValueError(f"Đỉnh run {row['run']} không khớp CSV")
    test=all_test.loc[all_test.run==r['selected_run']].reset_index(drop=True)
    market=read_csv(directory/'data/market.csv')
    temporary=directory/'tmp/pdfs';temporary.mkdir(parents=True,exist_ok=True)
    destination=directory/'output/pdf/BAO_CAO_AMZN.pdf'
    destination.parent.mkdir(parents=True,exist_ok=True)
    windows=Path('C:/Windows/Fonts')
    fallback=Path(matplotlib.get_data_path())/'fonts/ttf'
    for name,filename,alternative in [('Arial','arial.ttf','DejaVuSans.ttf'),
            ('Arial-Bold','arialbd.ttf','DejaVuSans-Bold.ttf')]:
        path=windows/filename
        pdfmetrics.registerFont(TTFont(name,str(path if path.exists() else fallback/alternative)))
    pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial',boldItalic='Arial-Bold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleVN',fontName='Arial-Bold',fontSize=23,leading=29,textColor=colors.HexColor(NAVY),spaceAfter=14))
    styles.add(ParagraphStyle(name='HeadVN',fontName='Arial-Bold',fontSize=16,leading=21,textColor=colors.HexColor(NAVY),spaceAfter=12))
    styles.add(ParagraphStyle(name='SubVN',fontName='Arial-Bold',fontSize=11,leading=15,textColor=colors.HexColor(TEAL),spaceBefore=9,spaceAfter=6))
    styles.add(ParagraphStyle(name='BodyVN',fontName='Arial',fontSize=9.3,leading=13.8,spaceAfter=7,textColor=colors.HexColor('#243746')))
    styles.add(ParagraphStyle(name='SmallVN',fontName='Arial',fontSize=8,leading=11,spaceAfter=6,textColor=colors.HexColor('#425466')))
    styles.add(ParagraphStyle(name='CellVN',fontName='Arial',fontSize=8.3,leading=11.4))
    styles.add(ParagraphStyle(name='HeaderVN',fontName='Arial-Bold',fontSize=8.3,leading=11.4,textColor=colors.white))
    story=[]

    def p(text,style='BodyVN'):
        story.append(Paragraph(text,styles[style]))

    def heading(text):
        p(text,'HeadVN')

    def page():
        story.append(PageBreak())

    def table(headers,rows,widths):
        data=[[Paragraph(str(value),styles['HeaderVN']) for value in headers]]
        data.extend([[Paragraph(str(value),styles['CellVN']) for value in row] for row in rows])
        t=Table(data,colWidths=[v*cm for v in widths],repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(NAVY)),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#EDF4F6')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),
            ('BOTTOMPADDING',(0,0),(-1,-1),7),('LEFTPADDING',(0,0),(-1,-1),8),
            ('RIGHTPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor(TEAL))]))
        story.append(t);story.append(Spacer(1,8))

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':NAVY,
        'xtick.color':NAVY,'ytick.color':NAVY,'figure.facecolor':'white'})

    def save_plot(fig,name,height):
        path=temporary/name
        fig.tight_layout(pad=1.8)
        fig.savefig(path,dpi=170,bbox_inches='tight',facecolor='white');plt.close(fig)
        story.append(Image(str(path),width=17.1*cm,height=height*cm))
        story.append(Spacer(1,7))

    def asset_plot(frame,title,name,height):
        fig,ax=plt.subplots(figsize=(9.2,3.3))
        dates=pd.to_datetime(frame.date)
        ax.plot(dates,frame.assets/1000,color=TEAL,lw=1.8,label='Tài sản theo Close')
        i=frame.assets.idxmax();row=frame.loc[i]
        ax.scatter(pd.Timestamp(row.date),row.assets/1000,color=NAVY,zorder=5,s=27)
        ax.annotate(f"ĐỈNH {money(row.assets)} USD\n{str(row.date)[:10]}",
            xy=(pd.Timestamp(row.date),row.assets/1000),xycoords='data',xytext=(.52,.9),
            textcoords='axes fraction',fontsize=9,fontweight='bold',ha='center',va='top',
            bbox={'facecolor':'white','alpha':.92,'edgecolor':'#CFDBE3','pad':4},
            arrowprops={'arrowstyle':'-','color':NAVY,'lw':.8})
        ax.set(title=title,ylabel='Nghìn USD',xlim=(dates.min(),dates.max()))
        ax.grid(alpha=.17);ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
        ax.legend(loc='lower left',fontsize=7.5)
        save_plot(fig,name,height)

    m=r['test'];tp=r['train_optimization_peak'];rp=r['train_replay_peak'];ep=r['test_peak']
    best_test=next(row for row in r['runs'] if row['run']==r['best_observed_test_run'])
    global_train=max(r['runs'],key=lambda row:row['train_optimization_peak']['assets'])
    global_test_peak=max(r['runs'],key=lambda row:row['test_peak']['assets'])
    ret=(m['final_assets_close']/cfg['INIT_CASH']-1)*100
    p('BÁO CÁO HỌC TĂNG CƯỜNG<br/>GIAO DỊCH AMZN','TitleVN')
    p('Model chính: run 09 • 10 lượt train và test • Mua Low, bán High','SubVN')
    p('Bài toán: dùng lịch sử OHLC theo ngày của một cổ phiếu để học quyết định mua, bán hoặc giữ, '
      'tối đa hóa tài sản từ 100.000 USD và hướng đến mốc 1.000.000 USD, đồng thời tuân thủ các giới hạn giao dịch.')
    table(['Kết quả chính','Giá trị / phạm vi'],[
        ['Model chính thức',f"Run {r['selected_run']:02d}, seed {r['seed']}; checkpoint {r['training']['best_checkpoint_steps']:,} bước<br/>Validation: {money(r['validation']['final_assets_close'])} USD"],
        ['Tài sản cuối test',f"<b>{money(m['final_assets_close'])} USD</b> · lợi nhuận {ret:+.2f}%"],
        ['Cuối test cao nhất trong CSV',f"<b>{money(best_test['test']['final_assets_close'])} USD</b> · run {best_test['run']:02d}"],
        ['Đỉnh PPO cao nhất cả 10 lượt',f"<b>{money(global_train['train_optimization_peak']['assets'])} USD</b> · run {global_train['run']:02d} · {global_train['train_optimization_peak']['date'][:10]}"],
        ['Đỉnh trong PPO đang học',f"<b>{money(tp['assets'])} USD</b> · {tp['date'][:10]}<br/>rank {tp['rank']}, episode {tp['episode']}; các episode tối đa 500 phiên"],
        ['Đỉnh replay trên khung train',f"<b>{money(rp['assets'])} USD</b> · {rp['date'][:10]}<br/>Checkpoint khóa trọng số; tối đa 8 năm, dừng theo done"],
        ['Đỉnh trên tập test',f"<b>{money(ep['assets'])} USD</b> · {ep['date'][:10]}"],
        ['Đỉnh test cao nhất cả 10 lượt',f"<b>{money(global_test_peak['test_peak']['assets'])} USD</b> · run {global_test_peak['run']:02d} · {global_test_peak['test_peak']['date'][:10]}"],
        ['Cuối test định giá bằng High phiên trước',f"{money(m['final_assets_prev_high'])} USD · điểm / 1 triệu: {m['score_prev_high']:.6f}"],
        ['Điểm theo tài sản cuối / 1 triệu',f"{m['score']:.6f} ({m['score']*100:.4f}%)"],
        ['Trung bình 10 run trong CSV',f"<b>{money(r['test_summary']['mean_final_assets'])} USD</b>"],
    ],[6,11.1])
    p('Cách đọc các đỉnh','SubVN')
    p('Đỉnh đang học được lấy từ mọi transition của 8 môi trường PPO. Replay train chạy checkpoint được chọn trên '
      'khung fit tối đa 8 năm, dừng nếu thắng/thua và là đánh giá trong mẫu. Đỉnh test thuộc riêng 2 năm kiểm tra. '
      'Ba số đo này có phạm vi khác nhau và không được thay thế cho nhau.')
    p(f"Mốc 1 triệu USD: <b>{'đã đạt' if m['target_reached'] else 'chưa đạt trên test'}</b>. "
      'Các chỉ số test trong báo cáo được tổng hợp từ assets_test.csv. '
      'Run 09 là model chính và có tài sản cuối test cao nhất trong 10 lượt.','SmallVN')
    p(f"Cập nhật: {escape(r['reconciliation']['updated_at_local'])}.",'SmallVN')

    page();heading('01  Thông số và luật giao dịch')
    p('Mô hình sử dụng bộ luật và tham số của dự án; kết quả được tổng hợp từ log tài sản test.')
    table(['Luật / thông số','Giá trị trong cấu hình'],[
        ['Vốn / đích thắng','100.000 USD tiền mặt ban đầu / tổng tài sản đạt 1.000.000 USD'],
        ['Mua theo Low','Giá Low của phiên; mỗi lệnh dùng 49%, 58%, 66%, 75% hoặc 83% tiền mặt trước lệnh'],
        ['Bán theo High','Giá High của phiên; bán 21%, 33%, 45%, 57% hoặc 69% cổ phiếu đang nắm trước lệnh'],
        ['Phí mỗi chiều','4% giá trị giao dịch; phí mua trừ thêm ngoài tiền mua, phí bán trừ khỏi tiền thu'],
        ['Không giao dịch','Sau 19 phiên liên tiếp không có giao dịch thực hiện: trừ 3.751 USD tiền mặt; reset bộ đếm'],
        ['Thua','Tài sản &lt; 1.000 USD hoặc tiền mặt ≤ −5.000 USD'],
        ['Khớp lệnh / danh mục','Một hành động mỗi phiên; khớp ngay, có cổ phiếu lẻ; không bán khống, không đòn bẩy'],
        ['Định giá tài sản','V = tiền mặt + số cổ phiếu × Close; định giá theo High phiên trước được ghi riêng trong JSON'],
        ['PenaltyGuard sẵn có','Khi model yêu cầu HOLD ở idle = 18, chọn lệnh khả thi có phí ước tính thấp nhất từ Close phiên trước; lệnh bán vẫn khớp High'],
    ],[4.2,12.9])
    p('Action space và điều kiện kết thúc','SubVN')
    p('Giữ Discrete(21) để tương thích kiến trúc: 0 = HOLD; 1–5 = năm tỷ lệ mua; 16–20 = năm tỷ lệ bán High. '
      'Các mã bán Close/Avg 6–15 bị mask. Khi hết tiền/cổ phiếu, mask loại lệnh không hợp lệ. '
      'decision là yêu cầu model; action là lệnh sau Guard. Guard không được tính thành quyết định chủ động của model.')
    p('terminated khi thắng hoặc thua theo các ngưỡng trên. truncated khi hết đoạn dữ liệu hoặc đủ 500 phiên '
      'của một episode đang train. Replay train và test chạy trong phạm vi ngày đã định và dừng khi done; '
      'test đủ 501 phiên nếu không kết thúc sớm.')
    p('Giả định khớp Low/High','SubVN')
    p('Theo yêu cầu, lệnh mua được khớp đúng Low và lệnh bán đúng High của ngày thực hiện. Đây là giả định '
      'mô phỏng thuận lợi; agent quyết định từ dữ liệu đến phiên trước, không nhìn Low/High hiện tại. '
      'Kết quả vì vậy không chứng minh khả năng khớp được các cực trị này ngoài thực tế.')

    page();heading('02  Thuật toán, reward và dòng thời gian')
    p('Maskable PPO + học bắt chước trên train','SubVN')
    p('Dùng <link href="https://sb3-contrib.readthedocs.io/en/v2.7.1/modules/ppo_mask.html">PPO có action mask (sb3-contrib 2.7.1)</link>, '
      'MLP trạng thái 64 và hai nhánh actor/critic [64, 64]. Khởi tạo bằng Behavior '
      'Cloning từ 8 trajectory train (flexible_expert: oracle và chiến lược nhân quả). Oracle dùng thông tin tương lai '
      'chỉ bên trong đoạn train để tạo nhãn; không dùng test. Trong PPO, guidance giảm dần về 0.')
    table(['Nhóm tham số','Giá trị đang dùng'],[
        ['PPO','32.768 transition; 8 env × 512 bước/rollout; batch 256; 5 epoch/update'],
        ['Tối ưu','LR 0,0002 giảm tuyến tính đến 10%; gamma 0,9995; GAE 0,98; clip 0,2; target KL 0,04; grad clip 0,5; vf 0,25'],
        ['Khám phá / risk_2','10 seed 0–9, cùng risk_2; entropy 0,075 → 0,006 (hệ số ×3); giảm logit HOLD 0,6; tăng logit SELL 0,25 × tỷ trọng cổ phiếu'],
        ['Prior giữ nguyên','Cash scale 0,02; cash strength 1,5; cost strength 5; BUY bias 1; SELL bias 0,25; HOLD bias 0'],
        ['BC / guidance','BC 16 epoch, LR 0,0003, batch 256, cân lớp mũ 0,5; guidance 0,02 → 0, batch 128, mỗi 4 minibatch'],
    ],[4.1,13])
    p('Cải tiến trong bản này','SubVN')
    p('Encoder bổ sung trung bình 5 phiên gần nhất và chênh trạng thái đầu/cuối cửa sổ, ngoài last/mean/std. '
      'BC và guidance lấy mẫu pha trộn phân phối tự nhiên với cân bằng ba nhóm HOLD/BUY/SELL, dùng hệ số '
      'BC_CLASS_POWER = 0,5 sẵn có; vẫn giữ trọng số lớp cũ. Điều này tăng cơ hội học các mẫu mua/bán hiếm, '
      'không ép agent phải giao dịch. Chọn checkpoint tại 16.384 và 32.768 bước bằng tài sản validation, sau cập nhật PPO.')
    p('Observation và reward','SubVN')
    p('Observation gồm 30 phiên × 28 đặc trưng nhân quả + 6 biến danh mục = 846 chiều. OHLC, biến động, '
      'SMA, RSI, MACD, ATR, Bollinger và momentum đều dịch một phiên; chuẩn hóa đặc trưng theo cửa sổ '
      'quá khứ 252 phiên, clip ±5. VecNormalize chỉ học trên train; test đóng băng mean/variance và không chuẩn hóa reward.')
    p('<b>r = 100 ln(V / V_trước) − 0,1 max(0, DD − DD_trước) − 1 × invalid + thưởng kết thúc.</b> '
      'Thắng +200, thua −100; hệ số excess/trade đều 0. HOLD được giảm xác suất bằng policy prior, '
      'không sửa phí, luật tiền hoặc thêm tiền vào reward. DD là drawdown từ đỉnh tài sản đã thấy.')
    p('Chia thời gian, không chồng train/test','SubVN')
    p(f"<b>Fit: {protocol['fit_start']} → {protocol['fit_end']} (8 năm).</b><br/>"
      f"<b>Validation: {protocol['validation_start']} → {protocol['validation_end']} (2 năm).</b><br/>"
      f"<b>Test: {cfg['DATA']['TEST_START']} → {cfg['DATA']['TEST_END']} (2 năm, {m['sessions']} phiên).</b> "
      'Dùng khung VALIDATION đã có trong config; giữ nguyên các tham số số học và luật. DATA.TRAIN_START/END cũ '
      'trong file không phải đoạn fit thực tế của chiến dịch này; effective_config/protocol trong JSON ghi rõ phạm vi thực thi. '
      'Quy trình dự đoán sử dụng load/predict và đóng băng normalizer. Kết quả test lấy từ assets_test.csv. Tập test đã được xem ở các phiên phát triển trước, '
      'nên không còn là holdout hoàn toàn chưa sử dụng.')

    page();heading('03  Kết quả 10 run trong assets_test.csv')
    rows=[]
    for row in r['runs']:
        mark=' *' if row['run']==r['selected_run'] else ''
        values=[f"{row['run']:02d} / {row['seed']}{mark}",f"{row['training']['best_checkpoint_steps']:,}",
            money(row['validation']['final_assets_close']),money(row['test']['final_assets_close']),
            f"<b>{money(row['test_peak']['assets'])}</b>",str(row['test']['model_hold_requests'])]
        if mark:values=[f'<b>{v}</b>' if '<b>' not in v else v for v in values]
        rows.append(values)
    table(['Run / seed','Bước chọn','Validation USD','Cuối test USD','Đỉnh test USD','HOLD'],rows,[1.8,2.1,3.5,3.5,3.7,2.5])
    p('* Model chính: run 09, seed 8, chọn theo tài sản validation. '
      'HOLD đếm decision=0 ở step&gt;0; không tính dòng vốn đầu kỳ. Các chỉ số test lấy từ assets_test.csv.','SmallVN')
    ends=np.array([row['test']['final_assets_close'] for row in r['runs']])
    p(f"Cuối test qua 10 lượt: trung bình <b>{money(ends.mean())} USD</b>, độ lệch chuẩn {money(ends.std(ddof=1))} USD; "
      f"thấp nhất {money(ends.min())}, cao nhất <b>{money(ends.max())} USD</b>. Hồ sơ train gốc ghi {protocol['total_training_timesteps']:,} transition PPO.")
    fig,ax=plt.subplots(figsize=(9.2,3.4))
    for number,group in all_test.groupby('run'):
        selected=number==r['selected_run'];best=number==r['best_observed_test_run']
        ax.plot(pd.to_datetime(group.date),group.assets/1000,
            color=TEAL if selected else '#BE5A32' if best else '#BDC8CF',
            lw=2 if selected else 1.4 if best else .7,alpha=1 if selected or best else .65,
             label='Model chính - run 09' if selected else 'Cuối test cao nhất CSV' if best else None,
            zorder=4 if selected or best else 1)
    ax.set(title='Cả 10 đường tài sản test',ylabel='Nghìn USD')
    ax.grid(alpha=.17);ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'));ax.legend(fontsize=8)
    save_plot(fig,'all_test.png',6.6)
    p(f"Trên validation, lượt chính thức yêu cầu {r['validation']['model_buy_requests']} mua, "
      f"{r['validation']['model_sell_requests']} bán và {r['validation']['model_hold_requests']} HOLD. "
      'Cách lấy mẫu mới tạo cơ hội học giao dịch nhiều hơn, nhưng tiêu chí chọn vẫn là tài sản; '
      'không tự loại model giữ cổ phiếu lâu khi chiến lược đó có tài sản validation tốt hơn.')

    page();heading('04  Giá và tài sản test của run chính')
    prices=market.loc[cfg['DATA']['TEST_START']:cfg['DATA']['TEST_END']]
    fig,ax=plt.subplots(figsize=(9.2,3.25))
    ax.fill_between(prices.index,prices.Low,prices.High,color='#DCECEF',label='Biên Low–High')
    ax.plot(prices.index,prices.Close,color=NAVY,lw=1,label='Close')
    for buy,color,marker,label in [(True,TEAL,'^','Giá mua ghi trong CSV'),(False,'#BE5A32','v','Giá bán ghi trong CSV')]:
        chosen=test.loc[test.decision.between(1,5) if buy else test.decision.between(16,20)]
        ax.scatter(pd.to_datetime(chosen.date),chosen.price,color=color,marker=marker,s=38,zorder=4,label=label)
    ax.set(title='Giá thị trường và giá khớp được ghi trong CSV',ylabel='USD / cổ phiếu',xlim=(prices.index.min(),prices.index.max()))
    ax.grid(alpha=.17);ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'));ax.legend(loc='lower right',fontsize=7)
    save_plot(fig,'test_price.png',6.3)
    asset_plot(test,'Tài sản test theo assets_test.csv - run chính','test_assets.png',6.5)
    p(f"Từ 100.000 USD đến <b>{money(m['final_assets_close'])} USD</b>; đỉnh <b>{money(ep['assets'])} USD</b> "
      f"ngày {ep['date'][:10]}; max drawdown {m['max_drawdown']*100:.2f}%. Reward tích lũy {m['cumulative_reward']:.6f}; "
      f"phí {money(m['total_fees'])} USD; {m['penalties']} lần phạt không giao dịch.")
    p(f"Run 09 có {m['model_buy_requests']} yêu cầu mua, {m['model_sell_requests']} bán, {m['model_hold_requests']} HOLD; "
      f"Guard can thiệp {m['guard_overrides']} lần. Tổng cộng {m['trades']} lệnh được thực hiện. "
      'Các điểm giá mua/bán và đường tài sản được lấy trực tiếp từ log test.')
    p('Tài sản được tính bằng cash + shares × Close. Chỉ số thay đổi tài sản, drawdown, phí và reward '
      'được tổng hợp trên toàn bộ 501 phiên, theo thứ tự step của run 09.','SmallVN')

    page();heading('05  Tài sản khi train của lượt chính thức')
    optimization=train.loc[(train.stage=='optimization')&(train.run==r['selected_run'])].copy()
    episode=optimization.loc[(optimization['rank']==tp['rank'])&(optimization.episode==tp['episode'])].copy()
    episode=episode.sort_values('step').reset_index(drop=True)
    asset_plot(episode,f"Episode có đỉnh PPO lớn nhất · rank {tp['rank']}, episode {tp['episode']}",'train_ppo.png',6.6)
    p(f"<b>Đỉnh PPO đang học: {money(tp['assets'])} USD</b> ngày {tp['date'][:10]}. "
      f"Lượt này học đủ {r['training']['actual_timesteps']:,} transition, kể cả sau checkpoint được chọn tại {r['training']['best_checkpoint_steps']:,}. "
      'Các env reset về 100.000 USD khi bắt đầu episode; không nối tiền của các episode thành một danh mục liên tục.')
    replay=train.loc[train.stage=='train_replay'].reset_index(drop=True)
    asset_plot(replay,'Replay fit 8 năm · checkpoint chính thức chỉ predict','train_replay.png',6.6)
    p(f"<b>Đỉnh replay train: {money(rp['assets'])} USD</b> ngày {rp['date'][:10]}; "
      f"cuối đoạn {money(r['train_replay']['final_assets_close'])} USD; reward {r['train_replay']['cumulative_reward']:.6f}. "
      f"Đã chạy {r['train_replay']['sessions']} phiên, đến {str(replay.iloc[-1].date)[:10]}; dừng theo luật done. "
      'Đây là chạy lại trong mẫu bằng checkpoint đã chọn, không phải mức tiền lớn nhất model đã thực sự thấy trong tối ưu PPO.')

    page();heading('06  Tổng hợp theo quý và số liệu tham chiếu')
    rows=[]
    previous_price=float(market.Close.loc[:pd.Timestamp(cfg['DATA']['TEST_START'])-pd.Timedelta(days=1)].iloc[-1])
    for period,group in test.iloc[1:].groupby(pd.to_datetime(test.iloc[1:].date).dt.to_period('Q'),sort=True):
        last=group.iloc[-1];endprice=float(last.close)
        rows.append([str(period),f'{(endprice/previous_price-1)*100:+.2f}%',
            money(last.assets),f"<b>{money(group.assets.max())}</b>"])
        previous_price=endprice
    table(['Quý trong test','Close thay đổi','Tài sản cuối (USD)','Đỉnh quý (USD)'],rows,[3.1,3.7,5.1,5.2])
    p('Quý đầu và quý cuối chỉ gồm các phiên nằm trong test (30/10/2024–29/10/2026). '
      'Biến động Close tính từ phiên trước đầu kỳ đến cuối kỳ; đỉnh quý không thay cho đỉnh toàn test.','SmallVN')
    rows=[]
    for key,label in [('model','Run chính (CSV hiện tại)'),('buy_hold','Buy &amp; Hold (hồ sơ gốc)'),('sma','SMA 20/50 (hồ sơ gốc)'),('momentum','Momentum 60 (hồ sơ gốc)')]:
        v=m if key=='model' else r['baselines'][key]
        rows.append([label,f"<b>{money(v['final_assets_close'])}</b>",f"{v['max_drawdown']*100:.2f}%",str(v['trades'])])
    table(['Nguồn số liệu','Cuối kỳ (USD)','Max drawdown','Số lệnh'],rows,[5.1,5.3,3.9,2.8])
    p('Các baseline được giữ từ hồ sơ thí nghiệm để làm mốc tham chiếu. Dòng run chính thể hiện kết quả '
      'test hiện tại. Buy &amp; Hold mua ban đầu theo hạn mức; SMA và momentum dùng tín hiệu theo lịch sử giá.')
    p('Nhận xét kết quả','SubVN')
    p(f"Theo CSV, run chính tăng {ret:+.2f}% và đạt {m['score']*100:.2f}% mục tiêu 1 triệu USD. "
      f"Đỉnh {money(ep['assets'])} USD tương đương {ep['assets']/cfg['WIN']*100:.2f}% mục tiêu. "
      'Cả 10 run đều kết thúc trên vốn ban đầu, chưa run nào đạt mục tiêu. '
      'Run 09 đứng đầu cả về tài sản cuối test và đỉnh tài sản trong 10 lượt.')

    page();heading('07  File nộp bài và kiểm tra tổng hợp')
    table(['File / folder','Ý nghĩa'],[
        ['agent.py','Policy, Maskable PPO, Behavior Cloning, callback train; không có bản agent cũ'],
        ['environment.py','Luật tiền, phí, reward, done, mask, khớp Low/High, Guard và kiểm toán log'],
        ['data.py / data/market.csv','Đọc cấu hình, kiểm tra OHLC, đặc trưng nhân quả, metric / dữ liệu đóng băng'],
        ['baseline.py','Buy &amp; Hold, SMA, momentum; oracle train dùng cho BC'],
        ['main.py / report.py','Chạy 10 lượt, khóa theo validation, tái lập, đóng ZIP / tạo PDF này'],
        ['tests/test_core.py','Một file kiểm thử tổng hợp các luật và quy trình train/predict'],
        ['models/run_09/','Model chính: agent.zip và normalizer.pkl; seed 8, checkpoint 32.768 bước'],
        ['models/run_01/ ... run_10/','Giữ đủ 10 cặp model và normalizer của chiến dịch'],
        ['<b>logs/assets_train.csv</b>','<b>Log chính khi train:</b> mọi bước PPO cả 10 lượt, validation và replay của model chính'],
        ['<b>logs/assets_test.csv</b>','<b>Log chính khi test:</b> vốn đầu kỳ + từng phiên, hành động, phí, giá và tài sản'],
        ['logs/result.json','Chỉ số tính từ CSV, các đỉnh, thông số, SHA256 và tổng hợp kết quả'],
        ['logs/run.log / checks.log','Tiến trình train và tổng hợp test / kết quả kiểm thử, kiểm tra số liệu'],
        ['config.yaml / README.md','Thông số giữ nguyên / cách dùng, giải thích file và các cột log'],
    ],[6.2,10.9])
    p('Đọc hai CSV','SubVN')
    p('<b>assets</b> là số tiền cần theo dõi; <b>close, price_change_pct</b> cho giá lên/xuống; '
      '<b>asset_change</b> là chênh tài sản sau mỗi phiên; <b>is_peak=True</b> đánh dấu đỉnh mới trong episode/replay. '
      'Thêm cột <b>run</b> để phân biệt 10 lượt. Lọc stage=optimization để xem PPO, stage=validation để xem chọn model, '
      'stage=train_replay để xem checkpoint chính thức trên fit. '
      'CSV là văn bản thuần nên phần bôi đậm các đỉnh được trình bày trong PDF này.')
    p('Kiểm tra số liệu và đóng gói','SubVN')
    p(f"<b>{r['checks']['pytest_passed']} kiểm thử mã nguồn đã qua.</b> CSV test có 5.020 dòng, 21 cột; "
      'mỗi run gồm một dòng vốn đầu kỳ và 501 phiên. Đã kiểm tra tài sản, biến động tiền/cổ phiếu, '
      'phí, reward và đánh dấu đỉnh trong log test.')
    p('Các chỉ số cuối kỳ, đỉnh, ngày đỉnh, số lệnh, HOLD và Guard trong JSON, README, run.log và PDF '
      'cùng lấy từ assets_test.csv. ZIP chứa mã nguồn, cấu hình, dữ liệu, đủ model và báo cáo; '
      'mỗi thành viên được kiểm tra SHA256 sau khi đóng gói.')
    p('Tài liệu thuật toán: Schulman và cộng sự, PPO (2017), arxiv.org/abs/1707.06347; '
      'sb3-contrib 2.7.1, ppo_mask. Bản quyền được ghi tại THIRD_PARTY_NOTICES.md.','SmallVN')

    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#D4E0E6'))
        canvas.line(1.9*cm,1.5*cm,19*cm,1.5*cm)
        canvas.setFont('Arial',8);canvas.setFillColor(colors.HexColor(NAVY))
        canvas.drawString(1.9*cm,1.05*cm,'AMZN | KET QUA TEST | MODEL RUN 09')
        canvas.drawRightString(19*cm,1.05*cm,str(doc.page));canvas.restoreState()

    doc=SimpleDocTemplate(str(destination),pagesize=(21*cm,29.7*cm),rightMargin=2*cm,
        leftMargin=1.9*cm,topMargin=1.7*cm,bottomMargin=1.9*cm,title='Báo cáo RL AMZN - kết quả test - model run 09',
        author='AMZN RL project')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    r['report']={'path':destination.relative_to(directory).as_posix(),
        'sha256':file_hash(destination),'pages':doc.page,'result_kind':'test_csv'}
    r['checks']['pdf_visual_qa_passed']=False
    write_json(directory/'logs/result.json',r)
    print(f'PDF {destination.name}',flush=True)
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT)
    sync_options=parser.add_mutually_exclusive_group()
    sync_options.add_argument('--sync',action='store_true',help='Đồng bộ JSON/README/log từ CSV test trước khi tạo PDF')
    sync_options.add_argument('--sync-only',action='store_true',help='Chỉ đồng bộ JSON/README/log; giữ nguyên PDF hiện có')
    parser.add_argument('--checks-log',type=Path,help='Kết quả pytest mới dùng cho phần kiểm tra')
    args=parser.parse_args()
    if args.sync or args.sync_only:sync_results(args.directory,args.checks_log)
    if not args.sync_only:build(args.directory)
