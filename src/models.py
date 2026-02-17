import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleFeatureAttentionNetwork(nn.Module):
    def __init__(self, input_dim=50, n_classes=12):
        super().__init__()
        # 1️⃣ Feature gating
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Hardsigmoid()
        )
        # 2️⃣ Multiscale CNN
        self.conv3 = nn.Conv1d(1, 32, 3, padding=1)
        self.conv5 = nn.Conv1d(1, 32, 5, padding=2)
        self.conv7 = nn.Conv1d(1, 32, 7, padding=3)
        self.bn = nn.BatchNorm1d(96)
        # 3️⃣ Cross-feature attention
        self.feature_attn = nn.MultiheadAttention(
            embed_dim=96,
            num_heads=4,
            batch_first=True
        )
        # 4️⃣ Attention pooling
        self.pool_attn = nn.Sequential(
            nn.Linear(96, 1),
            nn.Softmax(dim=1)
        )
        # 5️⃣ Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x, return_attention=False):
        if x.dim() == 3:
            x = x.squeeze(1)
        # Feature selection
        alpha = self.gate(x)             # (B, F)
        x = (x * alpha).unsqueeze(1)     # (B, 1, F)
        # CNN multiescala
        x = torch.cat([
            F.relu(self.conv3(x)),
            F.relu(self.conv5(x)),
            F.relu(self.conv7(x))
        ], dim=1)
        x = self.bn(x)                   # (B, 96, F)
        # Feature attention
        x = x.transpose(1, 2)            # (B, F, 96)
        x, attn_map = self.feature_attn(x, x, x)
        # Attention pooling
        weights = self.pool_attn(x)      # (B, F, 1)
        feat = (x * weights).sum(dim=1)  # (B, 96)
        logits = self.classifier(feat)
        if return_attention:
            return logits, alpha
        return logits
    
class SpectralBackbone(nn.Module):
    def __init__(self, input_dim=50):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Hardsigmoid()
        )
        self.conv3 = nn.Conv1d(1, 32, 3, padding=1)
        self.conv5 = nn.Conv1d(1, 32, 5, padding=2)
        self.conv7 = nn.Conv1d(1, 32, 7, padding=3)
        self.bn = nn.BatchNorm1d(96)
        self.feature_attn = nn.MultiheadAttention(
            embed_dim=96,
            num_heads=4,
            batch_first=True
        )
        self.pool_attn = nn.Sequential(
            nn.Linear(96, 1),
            nn.Softmax(dim=1)
        )
    def forward(self, x, return_attention=False):
        if x.dim() == 3:
            x = x.squeeze(1)
        alpha = self.gate(x)                 # (B, F)
        x = (x * alpha).unsqueeze(1)         # (B, 1, F)
        x = torch.cat([
            F.relu(self.conv3(x)),
            F.relu(self.conv5(x)),
            F.relu(self.conv7(x))
        ], dim=1)
        x = self.bn(x)                       # (B, 96, F)
        x = x.transpose(1, 2)                # (B, F, 96)
        x, attn_map = self.feature_attn(x, x, x)
        weights = self.pool_attn(x)          # (B, F, 1)
        feat = (x * weights).sum(dim=1)      # (B, 96)
        if return_attention:
            return feat, alpha, weights.squeeze(-1), attn_map
        return feat

class MultiScaleFeatureAttentionNetworkDual(nn.Module):
    def __init__(self, input_dim=50, n_classes=12):
        super().__init__()
        # Backbone compartido (MISMO para HG y LG)
        self.backbone = SpectralBackbone(input_dim)
        # Clasificador conjunto
        self.classifier = nn.Sequential(
            nn.Linear(96 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x_hg, x_lg, return_attention=False):
        if return_attention:
            feat_hg, alpha_hg, w_hg, att_hg = self.backbone(x_hg, True)
            feat_lg, alpha_lg, w_lg, att_lg = self.backbone(x_lg, True)
        else:
            feat_hg = self.backbone(x_hg)
            feat_lg = self.backbone(x_lg)
        feat = torch.cat([feat_hg, feat_lg], dim=1)  # (B, 192)
        logits = self.classifier(feat)
        if return_attention:
            return {
                "logits": logits,
                "hg": {"alpha": alpha_hg, "weights": w_hg, "attn": att_hg},
                "lg": {"alpha": alpha_lg, "weights": w_lg, "attn": att_lg},
            }
        return logits
    
class MSFAN_PoolingOnly(nn.Module):
    def __init__(self, input_dim=50, n_classes=12):
        super().__init__()
        # Projection to use the same pooling
        self.proj = nn.Linear(1, 96)
        # Attention pooling
        self.pool_attn = nn.Sequential(
            nn.Linear(96, 1),
            nn.Softmax(dim=1)
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)            # (B, F)
        x = self.proj(x.unsqueeze(-1))  # (B, F, 96)
        w = self.pool_attn(x)           # (B, F, 1)
        feat = (x * w).sum(dim=1)       # (B, 96)
        return self.classifier(feat)

class MSFAN_NoGating(nn.Module):
    def __init__(self, input_dim=50, n_classes=12):
        super().__init__()
        self.conv3 = nn.Conv1d(1, 32, 3, padding=1)
        self.conv5 = nn.Conv1d(1, 32, 5, padding=2)
        self.conv7 = nn.Conv1d(1, 32, 7, padding=3)
        self.bn = nn.BatchNorm1d(96)
        self.feature_attn = nn.MultiheadAttention(96, 4, batch_first=True)
        self.pool_attn = nn.Sequential(
            nn.Linear(96, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = x.unsqueeze(1)
        x = torch.cat([
            F.relu(self.conv3(x)),
            F.relu(self.conv5(x)),
            F.relu(self.conv7(x))
        ], dim=1)
        x = self.bn(x)
        x = x.transpose(1, 2)
        x, _ = self.feature_attn(x, x, x)
        w = self.pool_attn(x)
        feat = (x * w).sum(dim=1)
        return self.classifier(feat)
    
class MSFAN_NoConv(nn.Module):
    def __init__(self, input_dim=50, n_classes=12):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Hardsigmoid()
        )
        self.proj = nn.Linear(1, 96)
        self.feature_attn = nn.MultiheadAttention(96, 4, batch_first=True)
        self.pool_attn = nn.Sequential(
            nn.Linear(96, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x, return_attention=False):
        if x.dim() == 3:
            x = x.squeeze(1)
        alpha = self.gate(x)
        x = x * alpha                     # (B, F)
        x = x.unsqueeze(-1)               # (B, F, 1)
        x = self.proj(x)                  # (B, F, 96)
        x, _ = self.feature_attn(x, x, x)
        w = self.pool_attn(x)
        feat = (x * w).sum(dim=1)
        logits = self.classifier(feat)
        if return_attention:
            return logits, alpha
        return logits
    
class MSFAN_NoAttention(nn.Module):
    def __init__(self, input_dim=50, n_classes=12):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Hardsigmoid()
        )
        self.conv3 = nn.Conv1d(1, 32, 3, padding=1)
        self.conv5 = nn.Conv1d(1, 32, 5, padding=2)
        self.conv7 = nn.Conv1d(1, 32, 7, padding=3)
        self.bn = nn.BatchNorm1d(96)
        self.pool_attn = nn.Sequential(
            nn.Linear(96, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x, return_attention=False):
        if x.dim() == 3:
            x = x.squeeze(1)
        alpha = self.gate(x)
        x = (x * alpha).unsqueeze(1)
        x = torch.cat([
            F.relu(self.conv3(x)),
            F.relu(self.conv5(x)),
            F.relu(self.conv7(x))
        ], dim=1)
        x = self.bn(x).transpose(1, 2)
        w = self.pool_attn(x)
        feat = (x * w).sum(dim=1)
        logits = self.classifier(feat)
        if return_attention:
            return logits, alpha
        return logits
    
class MSFAN_AttentionOnly(nn.Module):
    def __init__(self, input_dim=50, n_classes=12):
        super().__init__()
        self.proj = nn.Linear(1, 96)
        self.feature_attn = nn.MultiheadAttention(96, 4, batch_first=True)
        self.pool_attn = nn.Sequential(
            nn.Linear(96, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = self.proj(x.unsqueeze(-1))  # (B, F, 96)
        x, _ = self.feature_attn(x, x, x)
        w = self.pool_attn(x)
        feat = (x * w).sum(dim=1)
        return self.classifier(feat)
    
class MSFAN_ConvOnly(nn.Module):
    def __init__(self, input_dim=50, n_classes=12):
        super().__init__()
        self.conv3 = nn.Conv1d(1, 32, 3, padding=1)
        self.conv5 = nn.Conv1d(1, 32, 5, padding=2)
        self.conv7 = nn.Conv1d(1, 32, 7, padding=3)
        self.bn = nn.BatchNorm1d(96)
        self.pool_attn = nn.Sequential(
            nn.Linear(96, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = x.unsqueeze(1)
        x = torch.cat([
            F.relu(self.conv3(x)),
            F.relu(self.conv5(x)),
            F.relu(self.conv7(x))
        ], dim=1)
        x = self.bn(x).transpose(1, 2)
        w = self.pool_attn(x)
        feat = (x * w).sum(dim=1)
        return self.classifier(feat)
    
class MSFAN_GatingOnly(nn.Module):
    def __init__(self, input_dim=50, n_classes=12):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Hardsigmoid()
        )
        self.proj = nn.Linear(1, 96)
        self.pool_attn = nn.Sequential(
            nn.Linear(96, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x, return_attention=False):
        if x.dim() == 3:
            x = x.squeeze(1)
        alpha = self.gate(x)
        x = self.proj((x * alpha).unsqueeze(-1))
        w = self.pool_attn(x)
        feat = (x * w).sum(dim=1)
        logits = self.classifier(feat)
        if return_attention:
            return logits, alpha
        return logits

class MultiScaleFeatureAttentionNetwork2(nn.Module):
# Same as MultiScaleFeatureAttentionNetwork, but returns alpha and weights for run_best_model.py
    def __init__(self, input_dim=50, n_classes=12):
        super().__init__()
        # 1. Feature gating
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Hardsigmoid()
        )
        # 2. Multiscale CNN
        self.conv3 = nn.Conv1d(1, 32, 3, padding=1)
        self.conv5 = nn.Conv1d(1, 32, 5, padding=2)
        self.conv7 = nn.Conv1d(1, 32, 7, padding=3)
        self.bn = nn.BatchNorm1d(96)
        # 3. Cross-feature attention
        self.feature_attn = nn.MultiheadAttention(
            embed_dim=96,
            num_heads=4,
            batch_first=True
        )
        # 4. Attention pooling
        self.pool_attn = nn.Sequential(
            nn.Linear(96, 1),
            nn.Softmax(dim=1)
        )
        # 5. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x, return_attention=False):
        if x.dim() == 3:
            x = x.squeeze(1)
        # Feature selection
        alpha = self.gate(x)             # (B, F)
        x = (x * alpha).unsqueeze(1)     # (B, 1, F)
        # Multiscale CNN
        x = torch.cat([
            F.relu(self.conv3(x)),
            F.relu(self.conv5(x)),
            F.relu(self.conv7(x))
        ], dim=1)
        x = self.bn(x)                   # (B, 96, F)
        # Feature attention
        x = x.transpose(1, 2)            # (B, F, 96)
        x, attn_map = self.feature_attn(x, x, x)
        # Attention pooling
        weights = self.pool_attn(x)      # (B, F, 1)
        feat = (x * weights).sum(dim=1)  # (B, 96)
        logits = self.classifier(feat)
        if return_attention:
            return logits, alpha, weights.squeeze(-1), attn_map
        return logits

# ==========================================
# 7. FACTORY MAESTRO
# ==========================================

def factory_models(architecture_name, mode, n_classes, device):

    model = None

    if architecture_name in ['MultiScaleFeatureAttentionNetwork', 'MultiScaleFeatureAttentionNetwork2', 'MSFAN_PoolingOnly',
                             'MSFAN_NoGating', 'MSFAN_NoConv', 'MSFAN_NoAttention',
                             'MSFAN_AttentionOnly', 'MSFAN_ConvOnly', 'MSFAN_GatingOnly']:
        target_dim = 100 if (mode == 'concat') else 50
        
        # Instance base model
        if architecture_name == 'MultiScaleFeatureAttentionNetwork': base = MultiScaleFeatureAttentionNetwork(target_dim, n_classes)
        elif architecture_name == 'MultiScaleFeatureAttentionNetwork2': base = MultiScaleFeatureAttentionNetwork2(target_dim, n_classes)
        elif architecture_name == 'MSFAN_PoolingOnly': base = MSFAN_PoolingOnly(target_dim, n_classes)
        elif architecture_name == 'MSFAN_NoGating': base = MSFAN_NoGating(target_dim, n_classes)
        elif architecture_name == 'MSFAN_NoConv': base = MSFAN_NoConv(target_dim, n_classes)
        elif architecture_name == 'MSFAN_NoAttention': base = MSFAN_NoAttention(target_dim, n_classes)
        elif architecture_name == 'MSFAN_AttentionOnly': base = MSFAN_AttentionOnly(target_dim, n_classes)
        elif architecture_name == 'MSFAN_ConvOnly': base = MSFAN_ConvOnly(target_dim, n_classes)
        elif architecture_name == 'MSFAN_GatingOnly': base = MSFAN_GatingOnly(target_dim, n_classes)

        # For Dual mode, use MultiScaleFeatureAttentionNetworkDual to accept (x_hg, x_lg)
        if mode == 'dual':
            model = MultiScaleFeatureAttentionNetworkDual(target_dim, n_classes)
        else:
            model = base

    else:
        raise ValueError(f"Architecture {architecture_name} not found.")

    return model.to(device)