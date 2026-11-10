#!/usr/bin/env python
"""读取 multiseed 结果 CSV(列: seed,ckpt,dsc,hd95)，输出 EMA vs last-epoch 的 mean±std 表。

用法:
    python aggregate_results.py <results.csv>

可随时单独运行（即使训练/测试还没全跑完）——它只读已写入 CSV 的行。
DSC 以百分比显示(×100)，HD95 用原值；std 为样本标准差(n-1)。
"""
import sys
import csv
import statistics as st
from collections import defaultdict


def ms(values, scale=1.0):
    """mean ± std 字符串；n=1 时只给单值。"""
    if not values:
        return 'n/a'
    v = [x * scale for x in values]
    if len(v) == 1:
        return f'{v[0]:.2f} (n=1)'
    return f'{st.mean(v):.2f} ± {st.stdev(v):.2f}'


def main(path):
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        print('找不到结果文件:', path)
        return
    if not rows:
        print('结果为空:', path)
        return

    grouped = defaultdict(lambda: {'dsc': [], 'hd95': []})   # 'EMA' / 'last-epoch' -> 指标列表
    by_key = defaultdict(dict)                                # (seed,run) -> {'last':(dsc,hd95), 'EMA':(...)}
    for r in rows:
        is_ema = r['ckpt'].startswith('ema')
        key = 'EMA' if is_ema else 'last-epoch'
        dsc, hd95 = float(r['dsc']), float(r['hd95'])
        grouped[key]['dsc'].append(dsc)
        grouped[key]['hd95'].append(hd95)
        by_key[(r['seed'], r.get('run', '-'))]['EMA' if is_ema else 'last'] = (dsc, hd95)

    print(f"\n== 汇总 (DSC 以 % 显示, HD95 原值; ± 为样本 std; n 含同 seed 重复) ==")
    print(f"{'ckpt':<12}{'n':<4}{'DSC mean±std':<22}{'HD95 mean±std':<22}")
    print('-' * 60)
    for key in ('last-epoch', 'EMA'):
        d, h = grouped[key]['dsc'], grouped[key]['hd95']
        print(f"{key:<12}{len(d):<4}{ms(d, 100):<22}{ms(h):<22}")

    print(f"\n== 逐 (seed,run) 明细 (DSC% / HD95) ==")
    print(f"{'seed':<8}{'run':<5}{'last-epoch':<18}{'EMA':<18}")
    for sk in sorted(by_key):
        l, e = by_key[sk].get('last'), by_key[sk].get('EMA')
        ls = f"{l[0]*100:.2f}/{l[1]:.2f}" if l else '-'
        es = f"{e[0]*100:.2f}/{e[1]:.2f}" if e else '-'
        print(f"{sk[0]:<8}{sk[1]:<5}{ls:<18}{es:<18}")
    print()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results.csv')
