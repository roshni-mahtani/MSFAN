import subprocess
import sys
import os
import argparse
from src.utils import summarize_ablation

BASE_COMMON_ARGS = [
    "--data_dir", "./data",
    "--batch_size", "128",
    "--lr", "0.0001",
    "--epochs", "100",
    "--reproducibility_seed", "42"
]

ABLATIONS = {

    "aggregation": {
        "results_dir": "./outputs/ABLATIONS/aggregation_ablation",
        "experiments": [
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "hg_only", "use_div_thick": "True", "lambda_sparse": 0.1},
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "hg_only", "use_div_thick": "False", "lambda_sparse": 0.1},
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "lg_only", "use_div_thick": "True", "lambda_sparse": 0.1},
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "lg_only", "use_div_thick": "False", "lambda_sparse": 0.1},
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "mean", "use_div_thick": "True", "lambda_sparse": 0.1},
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "mean", "use_div_thick": "False", "lambda_sparse": 0.1},
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "max", "use_div_thick": "True", "lambda_sparse": 0.1},
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "max", "use_div_thick": "False", "lambda_sparse": 0.1},
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "concat", "use_div_thick": "True", "lambda_sparse": 0.1},
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "concat", "use_div_thick": "False", "lambda_sparse": 0.1},
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "dual", "use_div_thick": "True", "lambda_sparse": 0.1},
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "dual", "use_div_thick": "False", "lambda_sparse": 0.1},
        ]
    },

    "modules": {
        "results_dir": "./outputs/ABLATIONS/modules_ablation",
        "experiments": [
            {"arch": "MSFAN_PoolingOnly", "mode": "hg_only", "use_div_thick": "True", "lambda_sparse": 0.1},
            {"arch": "MSFAN_GatingOnly", "mode": "hg_only", "use_div_thick": "True", "lambda_sparse": 0.1},
            {"arch": "MSFAN_ConvOnly", "mode": "hg_only", "use_div_thick": "True", "lambda_sparse": 0.1},
            {"arch": "MSFAN_AttentionOnly", "mode": "hg_only", "use_div_thick": "True", "lambda_sparse": 0.1},
            {"arch": "MSFAN_NoGating", "mode": "hg_only", "use_div_thick": "True", "lambda_sparse": 0.1},
            {"arch": "MSFAN_NoConv", "mode": "hg_only", "use_div_thick": "True", "lambda_sparse": 0.1},
            {"arch": "MSFAN_NoAttention", "mode": "hg_only", "use_div_thick": "True", "lambda_sparse": 0.1},
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "hg_only", "use_div_thick": "True", "lambda_sparse": 0.1},
        ]
    },

    "l1": {
        "results_dir": "./outputs/ABLATIONS/l1_ablation",
        "experiments": [
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "hg_only", "use_div_thick": "True", "lambda_sparse": 0.0},
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "hg_only", "use_div_thick": "True", "lambda_sparse": 0.1},
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "hg_only", "use_div_thick": "True", "lambda_sparse": 0.3},
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "hg_only", "use_div_thick": "True", "lambda_sparse": 0.5},
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "hg_only", "use_div_thick": "True", "lambda_sparse": 0.7},
            {"arch": "MultiScaleFeatureAttentionNetwork", "mode": "hg_only", "use_div_thick": "True", "lambda_sparse": 0.9},
        ]
    }
}


def run_experiment(exp, common_args, main_script_path):
    lambda_str = str(exp["lambda_sparse"]).replace('.', 'p')
    exp_name = f'{exp["arch"]}_{exp["mode"]}_{exp["use_div_thick"]}_lambda{lambda_str}'

    print(f"\n{'='*60}")
    print(f"🚀 STARTING EXPERIMENT: {exp_name}")
    print(f"{'='*60}\n")

    command = [
        sys.executable, main_script_path,
        "--name", exp_name,
        "--architecture", exp["arch"],
        "--mode", exp["mode"],
        "--use_div_thick", exp["use_div_thick"],
        "--lambda_sparse", str(exp["lambda_sparse"])
    ] + common_args

    try:
        subprocess.run(command, check=True)
        print(f"✅ EXPERIMENT COMPLETED")

    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR (code {e.returncode})")
        sys.exit(1)

    except KeyboardInterrupt:
        print("⚠️ Interrupted by user")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation",
        required=True,
        choices=ABLATIONS.keys(),
        help="Ablation study to run"
    )
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_script_path = os.path.join(current_dir, "src/main.py")

    if not os.path.exists(main_script_path):
        print("❌ main.py not found")
        return

    ablation_cfg = ABLATIONS[args.ablation]

    common_args = BASE_COMMON_ARGS + [
        "--results_dir", ablation_cfg["results_dir"]
    ]

    experiments_list = ablation_cfg["experiments"]

    print(f"\n📋 Running '{args.ablation}' ablation")
    print(f"📋 Results dir: {ablation_cfg['results_dir']}")
    print(f"📋 {len(experiments_list)} experiments queued\n")

    for i, exp in enumerate(experiments_list):
        print(f">>> Progress: {i+1}/{len(experiments_list)}")
        run_experiment(exp, common_args, main_script_path)
    
    summary_csv = os.path.join(ablation_cfg["results_dir"], "summary.csv")
    summarize_ablation(
        results_dir=ablation_cfg["results_dir"],
        experiments_list=ablation_cfg["experiments"],
        output_csv=summary_csv
    )

    print("\n🎉 ALL EXPERIMENTS COMPLETED")


if __name__ == "__main__":
    main()