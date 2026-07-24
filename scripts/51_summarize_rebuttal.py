import os
import json
import collections
import numpy as np

def main():
    json_path = 'rebuttal_preserve_reg_results.json'
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    wall_time = data.get('wall_time', 0.0)
    results = data.get('results', [])

    datasets = ["CIFAR-10", "FGVC-Aircraft", "Food-101"]
    lambdas = [0.01, 0.1, 1.0, 10.0]
    seeds = [42, 123, 456]

    print("=" * 90)
    print("DETAILED PER-RUN RESULTS (36 RUNS)")
    print("=" * 90)
    print(f"{'Dataset':<15} | {'Lambda':<8} | {'Seed':<6} | {'LP@1':<8} | {'AR@1':<8} | {'Displacement ||out-feats||':<25} | {'Checkpoint'}")
    print("-" * 110)

    for res in results:
        if res.get('status') == 'SUCCESS':
            print(f"{res['dataset']:<15} | {res['lambda_reg']:<8} | {res['seed']:<6} | {res['lp1']:.4f}   | {res['ar1']:.4f}   | {res['displacement']:.4f}                    | {res['ckpt_path']}")
        else:
            print(f"{res.get('dataset', 'Unknown'):<15} | {res.get('lambda_reg', 'N/A'):<8} | {res.get('seed', 'N/A'):<6} | FAILED: {res.get('error')}")

    print("\n" + "=" * 90)
    print("AGGREGATED STATISTICAL SUMMARY (MEAN ± STDERR OVER 3 SEEDS)")
    print("=" * 90)

    summary = collections.defaultdict(dict)
    worst_case_lp = {}

    for lam in lambdas:
        worst_lp = 1.0
        for dname in datasets:
            sub = [r for r in results if r['dataset'] == dname and r['lambda_reg'] == lam and r.get('status') == 'SUCCESS']
            lps = np.array([r['lp1'] for r in sub])
            ars = np.array([r['ar1'] for r in sub])
            disps = np.array([r['displacement'] for r in sub])

            mean_lp = np.mean(lps)
            stderr_lp = np.std(lps, ddof=1) / np.sqrt(len(lps)) if len(lps) > 1 else 0.0

            mean_ar = np.mean(ars)
            stderr_ar = np.std(ars, ddof=1) / np.sqrt(len(ars)) if len(ars) > 1 else 0.0

            mean_disp = np.mean(disps)

            summary[lam][dname] = {
                'lp_mean': mean_lp, 'lp_stderr': stderr_lp, 'lps': lps.tolist(),
                'ar_mean': mean_ar, 'ar_stderr': stderr_ar, 'ars': ars.tolist(),
                'disp_mean': mean_disp
            }
            if mean_lp < worst_lp:
                worst_lp = mean_lp

        worst_case_lp[lam] = worst_lp

    print(f"{'Lambda':<8} | {'Dataset':<15} | {'LP@1 (mean ± stderr)':<24} | {'AR@1 (mean ± stderr)':<24} | {'Displacement ||out-feats||':<25}")
    print("-" * 105)
    for lam in lambdas:
        for dname in datasets:
            info = summary[lam][dname]
            lp_str = f"{info['lp_mean']:.4f} ± {info['lp_stderr']:.4f}"
            ar_str = f"{info['ar_mean']:.4f} ± {info['ar_stderr']:.4f}"
            disp_str = f"{info['disp_mean']:.4f}"
            print(f"{lam:<8} | {dname:<15} | {lp_str:<24} | {ar_str:<24} | {disp_str:<25}")
        print(f"--> Worst-case LP@1 across datasets for lambda={lam}: {worst_case_lp[lam]:.4f}\n" + "-" * 105)

    print("\n" + "=" * 90)
    print("COMPACT MARKDOWN TABLE FOR NEURIPS REBUTTAL")
    print("=" * 90)

    md_lines = []
    md_lines.append("| $\\lambda_{\\text{reg}}$ | CIFAR-10 LP@1 | CIFAR-10 AR@1 | Aircraft LP@1 | Aircraft AR@1 | Food-101 LP@1 | Food-101 AR@1 | Worst LP@1 | Mean Displ. (CIFAR/Air/Food) |")
    md_lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for lam in lambdas:
        c10 = summary[lam]["CIFAR-10"]
        air = summary[lam]["FGVC-Aircraft"]
        food = summary[lam]["Food-101"]
        w_lp = worst_case_lp[lam]

        c10_lp = f"{c10['lp_mean']:.4f} ± {c10['lp_stderr']:.4f}"
        c10_ar = f"{c10['ar_mean']:.4f} ± {c10['ar_stderr']:.4f}"

        air_lp = f"{air['lp_mean']:.4f} ± {air['lp_stderr']:.4f}"
        air_ar = f"{air['ar_mean']:.4f} ± {air['ar_stderr']:.4f}"

        food_lp = f"{food['lp_mean']:.4f} ± {food['lp_stderr']:.4f}"
        food_ar = f"{food['ar_mean']:.4f} ± {food['ar_stderr']:.4f}"

        w_lp_str = f"**{w_lp:.4f}**"
        disps_str = f"{c10['disp_mean']:.3f} / {air['disp_mean']:.3f} / {food['disp_mean']:.3f}"

        row = f"| {lam} | {c10_lp} | {c10_ar} | {air_lp} | {air_ar} | {food_lp} | {food_ar} | {w_lp_str} | {disps_str} |"
        md_lines.append(row)

    print("\n".join(md_lines))

if __name__ == '__main__':
    main()
