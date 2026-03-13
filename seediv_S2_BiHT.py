import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from sklearn.metrics import accuracy_score

DATA_DIR = 'Submission_Dataset/SEED_IV'
BEST_MODEL_DIR = '../seediv_model'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BANDS = 5
NUM_CLASSES = 4
BATCH_SIZE = 128

SEED_IV_LABELS = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]

SEED_62_CHANNELS = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
    'O2', 'CB2'
]


def get_standard_10_20_coords():
    coords = {
        'FP1': [-0.309, 0.951, 0.0], 'FPZ': [0.0, 1.0, 0.0], 'FP2': [0.309, 0.951, 0.0],
        'AF3': [-0.4, 0.85, 0.35], 'AF4': [0.4, 0.85, 0.35],
        'F7': [-0.809, 0.588, -0.1], 'F5': [-0.7, 0.6, 0.25], 'F3': [-0.55, 0.65, 0.55], 'F1': [-0.28, 0.7, 0.65],
        'FZ': [0.0, 0.71, 0.71], 'F2': [0.28, 0.7, 0.65], 'F4': [0.55, 0.65, 0.55], 'F6': [0.7, 0.6, 0.25],
        'F8': [0.809, 0.588, -0.1],
        'FT7': [-0.9, 0.35, -0.2], 'FC5': [-0.8, 0.4, 0.3], 'FC3': [-0.6, 0.45, 0.65], 'FC1': [-0.3, 0.48, 0.8],
        'FCZ': [0.0, 0.5, 0.87], 'FC2': [0.3, 0.48, 0.8], 'FC4': [0.6, 0.45, 0.65], 'FC6': [0.8, 0.4, 0.3],
        'FT8': [0.9, 0.35, -0.2],
        'T7': [-0.951, 0.0, -0.309], 'C5': [-0.85, 0.0, 0.3], 'C3': [-0.71, 0.0, 0.71], 'C1': [-0.35, 0.0, 0.93],
        'CZ': [0.0, 0.0, 1.0], 'C2': [0.35, 0.0, 0.93], 'C4': [0.71, 0.0, 0.71], 'C6': [0.85, 0.0, 0.3],
        'T8': [0.951, 0.0, -0.309],
        'TP7': [-0.9, -0.35, -0.2], 'CP5': [-0.8, -0.4, 0.3], 'CP3': [-0.6, -0.45, 0.65], 'CP1': [-0.3, -0.48, 0.8],
        'CPZ': [0.0, -0.5, 0.87], 'CP2': [0.3, -0.48, 0.8], 'CP4': [0.6, -0.45, 0.65], 'CP6': [0.8, -0.4, 0.3],
        'TP8': [0.9, -0.35, -0.2],
        'P7': [-0.809, -0.588, -0.1], 'P5': [-0.7, -0.6, 0.25], 'P3': [-0.55, -0.65, 0.55], 'P1': [-0.28, -0.7, 0.65],
        'PZ': [0.0, -0.71, 0.71], 'P2': [0.28, -0.7, 0.65], 'P4': [0.55, -0.65, 0.55], 'P6': [0.7, -0.6, 0.25],
        'P8': [0.809, -0.588, -0.1],
        'PO7': [-0.7, -0.85, 0.0], 'PO5': [-0.55, -0.85, 0.3], 'PO3': [-0.3, -0.9, 0.4], 'POZ': [0.0, -0.92, 0.4],
        'PO4': [0.3, -0.9, 0.4], 'PO6': [0.55, -0.85, 0.3], 'PO8': [0.7, -0.85, 0.0],
        'CB1': [-0.5, -0.8, -0.5], 'O1': [-0.309, -0.951, 0.0], 'OZ': [0.0, -1.0, 0.0], 'O2': [0.309, -0.951, 0.0],
        'CB2': [0.5, -0.8, -0.5]
    }
    return coords


def get_10_regions_mapping():
    return {
        'M_Ant': ['FPZ', 'FZ', 'FCZ', 'CZ'], 'M_Post': ['CPZ', 'PZ', 'POZ', 'OZ'],
        'L_PF': ['FP1', 'AF3'], 'L_F': ['F7', 'F5', 'F3', 'F1', 'FC5', 'FC3', 'FC1', 'FT7'],
        'L_T': ['T7', 'C5', 'C3', 'C1', 'TP7', 'CP5', 'CP3', 'CP1'],
        'L_PO': ['P7', 'P5', 'P3', 'P1', 'PO7', 'PO5', 'PO3', 'CB1', 'O1'],
        'R_PF': ['FP2', 'AF4'], 'R_F': ['F8', 'F6', 'F4', 'F2', 'FC6', 'FC4', 'FC2', 'FT8'],
        'R_T': ['T8', 'C6', 'C4', 'C2', 'TP8', 'CP6', 'CP4', 'CP2'],
        'R_PO': ['P8', 'P6', 'P4', 'P2', 'PO8', 'PO6', 'PO4', 'CB2', 'O2']
    }


class FrequencyAttention(nn.Module):
    def __init__(self, num_bands=5, output_dim=32):
        super().__init__()
        self.attn_fc = nn.Sequential(nn.Linear(num_bands, 16), nn.Tanh(), nn.Linear(16, num_bands), nn.Softmax(dim=-1))
        self.feature_mapper = nn.Linear(num_bands, output_dim)

    def forward(self, x): return self.feature_mapper(x * self.attn_fc(x))


class SpatialRegionProjector(nn.Module):
    def __init__(self, in_channels_indices, channel_coords, input_dim=32, output_dim=64):
        super().__init__()
        self.indices = in_channels_indices
        region_coords = [channel_coords.get(SEED_62_CHANNELS[idx], [0, 0, 0]) for idx in in_channels_indices]
        self.register_buffer('coords', torch.tensor(region_coords, dtype=torch.float32))
        self.coord_encoder = nn.Linear(3, input_dim)
        self.aggregator = nn.Sequential(nn.Linear(len(in_channels_indices) * input_dim, 128), nn.LayerNorm(128),
                                        nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, output_dim))

    def forward(self, x_all_channels):
        x_region = x_all_channels[:, self.indices, :] + self.coord_encoder(self.coords).unsqueeze(0)
        return self.aggregator(x_region.flatten(1))


class DistanceAwareAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, distance_matrix, dropout=0.3):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('dist_matrix', torch.tensor(distance_matrix, dtype=torch.float32))
        self.lambda_decay = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        attn_bias = -1.0 * torch.abs(self.lambda_decay) * self.dist_matrix
        attn_out, _ = self.mha(x, x, x, attn_mask=attn_bias)
        return self.norm(x + self.dropout(attn_out))


class BroadcastCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.3):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.gate_fc = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_query, x_context):
        attn_out, _ = self.mha(query=x_query, key=x_context, value=x_context)
        concat = torch.cat([x_query, self.dropout(attn_out)], dim=-1)
        gamma = torch.sigmoid(self.gate_fc(concat))
        return self.norm((1 - gamma) * x_query + gamma * self.dropout(attn_out))


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x): return self.layer(x)


class S2_BiHT_V2_DANN(nn.Module):
    def __init__(self, num_classes=4, embed_dim=64, num_heads=4):
        super().__init__()
        coords_dict, region_map = get_standard_10_20_coords(), get_10_regions_mapping()
        channel_to_idx = {ch: i for i, ch in enumerate(SEED_62_CHANNELS)}
        self.freq_attn = FrequencyAttention(num_bands=N_BANDS, output_dim=32)
        self.left_projectors, self.right_projectors, self.mid_projectors = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        left_centers, right_centers, mid_centers = [], [], []

        def build_proj(names, proj_list, center_list):
            for name in names:
                ch_names = region_map[name]
                proj_list.append(
                    SpatialRegionProjector([channel_to_idx[ch] for ch in ch_names], coords_dict, input_dim=32,
                                           output_dim=embed_dim))
                center_list.append(np.mean(np.array([coords_dict[ch] for ch in ch_names]), axis=0))

        build_proj(['L_PF', 'L_F', 'L_T', 'L_PO'], self.left_projectors, left_centers)
        build_proj(['R_PF', 'R_F', 'R_T', 'R_PO'], self.right_projectors, right_centers)
        build_proj(['M_Ant', 'M_Post'], self.mid_projectors, mid_centers)

        def calc_dist(c): return np.array([[np.linalg.norm(c[i] - c[j]) for j in range(len(c))] for i in range(len(c))])

        self.left_self_attn = DistanceAwareAttention(embed_dim, num_heads, calc_dist(left_centers))
        self.right_self_attn = DistanceAwareAttention(embed_dim, num_heads, calc_dist(right_centers))
        self.mid_self_attn = DistanceAwareAttention(embed_dim, num_heads, calc_dist(mid_centers))
        self.l_cross_attn = BroadcastCrossAttention(embed_dim, num_heads)
        self.r_cross_attn = BroadcastCrossAttention(embed_dim, num_heads)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(14 * embed_dim, 128), nn.ReLU(), nn.Dropout(0.4),
                                        nn.Linear(128, num_classes))

        self.disc_left = DomainDiscriminator(4 * embed_dim)
        self.disc_right = DomainDiscriminator(4 * embed_dim)
        self.disc_diff = DomainDiscriminator(4 * embed_dim)

    def forward(self, x, alpha=None):
        x_spec = self.freq_attn(x.view(x.size(0), 62, 5))
        h_left = torch.stack([proj(x_spec) for proj in self.left_projectors], dim=1)
        h_right = torch.stack([proj(x_spec) for proj in self.right_projectors], dim=1)
        h_mid = torch.stack([proj(x_spec) for proj in self.mid_projectors], dim=1)

        h_left = self.left_self_attn(h_left)
        h_right = self.right_self_attn(h_right)
        h_mid = self.mid_self_attn(h_mid)

        ctx_l = torch.cat([h_right, h_mid], dim=1)
        ctx_r = torch.cat([h_left, h_mid], dim=1)

        h_left_final = self.l_cross_attn(x_query=h_left, x_context=ctx_l)
        h_right_final = self.r_cross_attn(x_query=h_right, x_context=ctx_r)

        h_diff = h_left_final - h_right_final
        h_concat = torch.cat([h_left_final, h_right_final, h_mid, h_diff], dim=1)

        class_out = self.classifier(h_concat)
        return class_out, None


class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def apply_session_zscore(trial_list):
    if not trial_list: return []
    full = np.concatenate(trial_list, axis=1)
    mu, std = np.mean(full, axis=1, keepdims=True), np.std(full, axis=1, keepdims=True)
    return [(t - mu) / (std + 1e-6) for t in trial_list]


def get_test_loader_for_subject(sid):
    sub_mat = glob.glob(os.path.join(DATA_DIR, f'{sid}_*.mat'))
    if not sub_mat: return None

    mat = sio.loadmat(sub_mat[0])
    test_X, test_y = [], []

    raw_trials = [mat[f'de_LDS{k}'] for k in range(1, 25) if f'de_LDS{k}' in mat]
    if not raw_trials: return None

    norm_trials = apply_session_zscore(raw_trials)

    for t_d, t_l in zip(norm_trials, SEED_IV_LABELS[:len(norm_trials)]):
        trans = t_d.transpose(1, 0, 2)
        if trans.shape[0] > 0:
            test_X.append(trans.reshape(trans.shape[0], -1))
            test_y.append(np.full(trans.shape[0], t_l))

    return DataLoader(EEGDataset(np.concatenate(test_X, axis=0), np.concatenate(test_y, axis=0)), batch_size=BATCH_SIZE,
                      shuffle=False)


if __name__ == '__main__':
    print("=" * 60)
    print("S2-BiHT (SEED-IV) Evaluation Script")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset path not found: {DATA_DIR}")
        exit()
    if not os.path.exists(BEST_MODEL_DIR):
        print(f"Error: Model weights path not found: {BEST_MODEL_DIR}")
        exit()

    model = S2_BiHT_V2_DANN(num_classes=NUM_CLASSES).to(DEVICE)
    all_subject_accs = []

    print("\nEvaluating 15 subjects from SEED-IV dataset...\n")

    for sid in range(1, 16):
        pth_path = os.path.join(BEST_MODEL_DIR, f'model_sub{sid}.pth')

        if not os.path.exists(pth_path):
            print(f"Subject {sid}: Model weight file not found, skipping.")
            continue

        model.load_state_dict(torch.load(pth_path, map_location=DEVICE), strict=False)
        model.eval()

        loader = get_test_loader_for_subject(sid)
        if not loader:
            print(f"Subject {sid}: Data file not found or corrupted, skipping.")
            continue

        preds, trues = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(DEVICE)
                out, _ = model(x, alpha=None)
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                trues.extend(y.numpy())

        acc = accuracy_score(trues, preds)
        all_subject_accs.append(acc)
        print(f"Subject {sid:02d} Accuracy: {acc * 100:.2f}%")

    if all_subject_accs:
        print("\n" + "=" * 40)
        print(f"Final Results (n={len(all_subject_accs)})")
        print(f"Average Accuracy: {np.mean(all_subject_accs) * 100:.2f}% +- {np.std(all_subject_accs) * 100:.2f}%")
        print("=" * 40)
    else:
        print("No subjects were evaluated. Please check file paths.")