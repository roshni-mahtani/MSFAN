import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses import SparseCrossEntropyLoss

from data_loader import load_all_data, get_fold_loaders
from models import factory_models
from utils import calculate_metrics, save_confusion_matrix, plot_loss, save_results_to_file, set_seed, check_gpu, str2bool, save_run_config

def train_one_fold(fold_idx, all_data, args, device):
    print(f"\n--- Starting Fold {fold_idx} ---")
    
    # Fast Loader (Data is prepared as tensors)
    train_loader, val_loader = get_fold_loaders(all_data, fold_idx, args)
    n_classes = len(all_data['classes'])
    
    model = factory_models(args.architecture, args.mode, n_classes, device)
    
    # Baseline setup
    criterion = SparseCrossEntropyLoss(lambda_sparse=args.lambda_sparse)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=False)
    
    train_losses, val_losses = [], []
    
    for epoch in range(args.epochs):
        
        # Training
        model.train()
        run_loss = 0.0
        
        for batch in train_loader:
            # Unpack depending on the mode (dual vs rest)
            if args.mode == 'dual':
                x_hg, x_lg, y = batch
                x_hg = x_hg.to(device, non_blocking=True)
                x_lg = x_lg.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                out = model(x_hg, x_lg, return_attention=True)
                logits = out["logits"]
                alpha_hg = out["hg"]["alpha"]
                alpha_lg = out["lg"]["alpha"]
                alpha = 0.5 * (alpha_hg + alpha_lg)
                outputs = (logits, alpha)
            else:
                x, y = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                try:
                    outputs = model(x, return_attention = True)
                except (TypeError, UnboundLocalError):
                    outputs = model(x)

            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            
        avg_train_loss = run_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                if args.mode == 'dual':
                    x_hg, x_lg, y = batch
                    x_hg = x_hg.to(device, non_blocking=True)
                    x_lg = x_lg.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    out = model(x_hg, x_lg, return_attention=True)
                    logits = out["logits"]
                    alpha_hg = out["hg"]["alpha"]
                    alpha_lg = out["lg"]["alpha"]
                    alpha = 0.5 * (alpha_hg + alpha_lg)
                    outputs = (logits, alpha)
                else:
                    x, y = batch
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    try:
                        logits, alpha = model(x, return_attention=True)
                        outputs = (logits, alpha)
                    except (TypeError, UnboundLocalError):
                        outputs = model(x)
                        logits = outputs

                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = correct / total
        
        scheduler.step(avg_val_loss)
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # === FINAL Evaluation: last epoch model ===
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in val_loader:
            if args.mode == 'dual':
                x_hg, x_lg, y = batch
                x_hg = x_hg.to(device)
                x_lg = x_lg.to(device)
                y = y.to(device)
                out = model(x_hg, x_lg, return_attention=True)
                logits = out["logits"]
                alpha_hg = out["hg"]["alpha"]
                alpha_lg = out["lg"]["alpha"]
                alpha = 0.5 * (alpha_hg + alpha_lg)
                outputs = (logits, alpha)
            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                try:
                    logits, alpha = model(x, return_attention=True)
                    outputs = (logits, alpha)
                except (TypeError, UnboundLocalError):
                    outputs = model(x)
                    logits = outputs
                
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    metrics, cm = calculate_metrics(np.array(all_targets), np.array(all_preds), np.array(all_probs), n_classes)
    return metrics, cm, train_losses, val_losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./outputs')
    parser.add_argument('--name', type=str, default='MSFAN_ABLATION')
    parser.add_argument('--data_dir', type=str, default='./data')
    
    parser.add_argument('--architecture', type=str, default='MultiScaleFeatureAttentionNetwork')
    parser.add_argument('--mode', type=str, default='hg_only', 
                        choices=['hg_only', 'lg_only', 'mean', 'max', 'concat', 'dual'])
    
    parser.add_argument('--use_div_thick', type=str2bool, default=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--reproducibility_seed', type=int, default=42)

    parser.add_argument('--lambda_sparse', type=float, default=1e-3)
    
    args = parser.parse_args()
    set_seed(args.reproducibility_seed)
    device = check_gpu()
    
    base_dir = os.path.join(args.results_dir, args.name)
    os.makedirs(os.path.join(base_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'plots'), exist_ok=True)
    save_run_config(args, base_dir)
    
    all_data = load_all_data(args)
    
    fold_metrics_list = []
    global_cm = None
    
    for fold in range(1, 6):
        metrics, cm, t_loss, v_loss = train_one_fold(fold, all_data, args, device)
        fold_metrics_list.append(metrics)
        if global_cm is None: global_cm = cm
        else: global_cm += cm
        plot_loss(t_loss, v_loss, os.path.join(base_dir, 'plots', f'loss_fold_{fold}.png'))
    
    classes_labels = [str(c) for c in all_data['classes']]
    save_confusion_matrix(global_cm, classes_labels, os.path.join(base_dir, 'confusion_matrix_global.png'))
    
    keys = fold_metrics_list[0].keys()
    avg_metrics = {k: np.mean([m[k] for m in fold_metrics_list]) for k in keys}
    std_metrics = {k: np.std([m[k] for m in fold_metrics_list]) for k in keys}
    save_results_to_file(fold_metrics_list, avg_metrics, std_metrics, os.path.join(base_dir, 'metrics'))
    
    print(f"\nCompleted. Global Acc: {avg_metrics['ACC']:.4f}")

if __name__ == "__main__":
    main()