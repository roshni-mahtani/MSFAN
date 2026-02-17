import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

class FastTensorDataLoader:
    """
    A tensor-optimized data loader that avoids the overhead of PyTorch's standard DataLoader.
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate number of batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
        
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def load_all_data(args):
    """
    Loads CSV data once and returns it in a dictionary with numpy arrays.
    """
    prefix = "div_thick" if args.use_div_thick else "balanced"
    hg_filename = f"HG_{prefix}_with_fold_class.csv"
    lg_filename = f"LG_{prefix}_with_fold_class.csv"
    
    hg_path = os.path.join(args.data_dir, hg_filename)
    lg_path = os.path.join(args.data_dir, lg_filename)
    
    print(f"[Data] Loading data from:\n - {hg_filename}\n - {lg_filename}")
    
    if not os.path.exists(hg_path) or not os.path.exists(lg_path):
        raise FileNotFoundError(f"Data files not found in {args.data_dir}")

    df_hg = pd.read_csv(hg_path)
    df_lg = pd.read_csv(lg_path)
    
    if not df_hg['measurement'].equals(df_lg['measurement']):
        raise ValueError("Error: HG and LG are not aligned.")
    
    meta_cols = ['measurement', 'fold', 'class']
    feat_cols = [c for c in df_hg.columns if c not in meta_cols]
    
    X_hg_all = df_hg[feat_cols].values.astype(np.float32)
    X_lg_all = df_lg[feat_cols].values.astype(np.float32)
    y_all = df_hg['class'].values
    folds_all = df_hg['fold'].values
    
    classes = sorted(list(set(y_all)))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    print(f"[Data] Loaded {len(X_hg_all)} samples. Shape: {X_hg_all.shape}. Classes: {len(classes)}.")
    
    return {
        'X_hg': X_hg_all, 'X_lg': X_lg_all, 'y': y_all, 
        'folds': folds_all, 'classes': classes, 'class_to_idx': class_to_idx
    }

def get_fold_loaders(all_data, fold_idx, args):
    """
    Prepares data for the fold and returns FastTensorDataLoaders.
    Performs all tensor manipulation (concat, unsqueeze) HERE 
    to avoid doing it in the training loop.
    """
    X_hg = all_data['X_hg']
    X_lg = all_data['X_lg']
    y = all_data['y']
    folds = all_data['folds']
    
    # Split
    train_mask = (folds != fold_idx)
    val_mask = (folds == fold_idx)
    
    X_hg_train, X_hg_val = X_hg[train_mask], X_hg[val_mask]
    X_lg_train, X_lg_val = X_lg[train_mask], X_lg[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    
    # Normalization
    scaler_hg = MinMaxScaler()
    X_hg_train = scaler_hg.fit_transform(X_hg_train)
    X_hg_val = scaler_hg.transform(X_hg_val)
    
    scaler_lg = MinMaxScaler()
    X_lg_train = scaler_lg.fit_transform(X_lg_train)
    X_lg_val = scaler_lg.transform(X_lg_val)

    # Convert to tensors
    t_hg_train = torch.from_numpy(X_hg_train).float()
    t_hg_val = torch.from_numpy(X_hg_val).float()
    t_lg_train = torch.from_numpy(X_lg_train).float()
    t_lg_val = torch.from_numpy(X_lg_val).float()
    
    t_y_train = torch.tensor([all_data['class_to_idx'][lbl] for lbl in y_train], dtype=torch.long)
    t_y_val = torch.tensor([all_data['class_to_idx'][lbl] for lbl in y_val], dtype=torch.long)

    # Prepare inputs according to mode
    # Goal: return (Input, Label) or (Input1, Input2, Label) ready for the network
    
    if args.mode == 'dual':
        # Add channel dimension: (N, 50) -> (N, 1, 50)
        train_inputs = (t_hg_train.unsqueeze(1), t_lg_train.unsqueeze(1))
        val_inputs = (t_hg_val.unsqueeze(1), t_lg_val.unsqueeze(1))
        
    elif args.mode == 'concat':
        # Stack along dim 1: (N, 2, 50)
        train_combined = torch.concatenate([t_hg_train, t_lg_train], dim=1)
        val_combined = torch.concatenate([t_hg_val, t_lg_val], dim=1)
        train_inputs = (train_combined.unsqueeze(1),)
        val_inputs = (val_combined.unsqueeze(1),)
        
    elif args.mode == 'mean':
        # Mean: (N, 1, 50)
        train_combined = ((t_hg_train + t_lg_train) / 2.0).unsqueeze(1)
        val_combined = ((t_hg_val + t_lg_val) / 2.0).unsqueeze(1)
        train_inputs = (train_combined,)
        val_inputs = (val_combined,)
        
    elif args.mode == 'max':
        # Max: (N, 1, 50)
        train_combined = torch.max(t_hg_train, t_lg_train).unsqueeze(1)
        val_combined = torch.max(t_hg_val, t_lg_val).unsqueeze(1)
        train_inputs = (train_combined,)
        val_inputs = (val_combined,)
        
    elif args.mode == 'hg_only':
        train_inputs = (t_hg_train.unsqueeze(1),)
        val_inputs = (t_hg_val.unsqueeze(1),)
        
    elif args.mode == 'lg_only':
        train_inputs = (t_lg_train.unsqueeze(1),)
        val_inputs = (t_lg_val.unsqueeze(1),)
    
    # Create FastLoaders
    # Note: *train_inputs unpacks the tuple (whether 1 or 2 tensors)
    train_loader = FastTensorDataLoader(*train_inputs, t_y_train, batch_size=args.batch_size, shuffle=True)
    val_loader = FastTensorDataLoader(*val_inputs, t_y_val, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, val_loader