import os
import re
import random
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# GENERAL UTILS

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def check_gpu():
    if torch.cuda.is_available():
        print(f"[SYSTEM] GPU: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    return torch.device('cpu')

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def save_run_config(args, output_dir):
    with open(os.path.join(output_dir, 'config_log.txt'), 'w') as f:
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg}: {value}\n")

# METRICS UTILS

def calculate_metrics(y_true, y_pred, y_prob, n_classes):
    cm = confusion_matrix(y_true, y_pred)
    sen, spe, npv, ppv = [], [], [], []
    
    # One-vs-Rest metrics
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        sen.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        spe.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        ppv.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        npv.append(tn / (tn + fn) if (tn + fn) > 0 else 0)
        
    acc = np.trace(cm) / np.sum(cm)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    # Macro AUC
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            y_onehot = label_binarize(y_true, classes=range(n_classes))
            auc = roc_auc_score(y_onehot, y_prob, multi_class='ovr', average='macro')
    except:
        auc = 0.0
        
    return {
        'ACC': acc,
        'SEN': np.mean(sen),
        'SPE': np.mean(spe),
        'PPV': np.mean(ppv),
        'NPV': np.mean(npv),
        'F1': f1,
        'AUC': auc
    }, cm

def save_confusion_matrix(cm, classes, path, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_loss(train_losses, val_losses, path):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.savefig(path)
    plt.close()

def save_results_to_file(metrics_list, global_metrics, fold_metrics_std, output_dir):
    with open(os.path.join(output_dir, 'metrics_report.txt'), 'w') as f:
        f.write("=== PER FOLD METRICS ===\n")
        for i, m in enumerate(metrics_list):
            f.write(f"Fold {i+1}: {m}\n")
        
        f.write("\n=== GLOBAL METRICS (Mean ± Std) ===\n")
        for k, v in global_metrics.items():
            f.write(f"{k}: {v:.4f} ± {fold_metrics_std[k]:.4f}\n")

# ABLATION TABLES UTILS

METRIC_PATTERN = re.compile(r"^(ACC|SEN|SPE|PPV|NPV|F1|AUC):\s+([0-9.]+)\s±\s([0-9.]+)")

def parse_metrics_report(report_path):
    metrics = {}
    with open(report_path, "r") as f:
        for line in f:
            match = METRIC_PATTERN.match(line.strip())
            if match:
                metric = match.group(1)
                mean = float(match.group(2))
                std = float(match.group(3))
                metrics[metric] = f"{mean:.4f} ± {std:.4f}"
    return metrics

def experiment_name_from_dict(exp_dict):
    lambda_str = str(exp_dict["lambda_sparse"]).replace(".", "p")
    return f"{exp_dict['arch']}_{exp_dict['mode']}_{exp_dict['use_div_thick']}_lambda{lambda_str}"

def summarize_ablation(results_dir, experiments_list, output_csv):
    rows = []
    for exp in experiments_list:
        exp_name = experiment_name_from_dict(exp)
        exp_path = os.path.join(results_dir, exp_name)
        report_path = os.path.join(exp_path, "metrics", "metrics_report.txt")
        if not os.path.exists(report_path):
            print(f"⚠️ metrics_report.txt not found for {exp_name}")
            continue
        metrics = parse_metrics_report(report_path)
        row = {
            "experiment": exp_name,
            **metrics
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Summary saved to: {output_csv}")