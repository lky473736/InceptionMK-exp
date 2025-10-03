import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from scipy.spatial.transform import Rotation
from thop import profile
from IPython.display import display
from mpl_toolkits.mplot3d import Axes3D

def load_uci_har_raw(dataset_path):
    SIGNALS = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]
    X_train_list, X_test_list = [], []
    for signal in SIGNALS:
        train_file = os.path.join(dataset_path, f"{signal}_train.txt")
        test_file = os.path.join(dataset_path, f"{signal}_test.txt")
        try:
            X_train_list.append(np.loadtxt(train_file, dtype=np.float32))
            X_test_list.append(np.loadtxt(test_file, dtype=np.float32))
        except Exception as e:
            print(f"Error loading file: {e}. Check dataset_path/UCI HAR Dataset.")
            return None, None, None, None, None
    if not X_train_list: return None, None, None, None, None
    X_train = np.transpose(np.stack(X_train_list, axis=-1), (0, 2, 1))
    X_test = np.transpose(np.stack(X_test_list, axis=-1), (0, 2, 1))
    try:
        y_train = np.loadtxt(os.path.join(dataset_path,"y_train.txt"), dtype=int) - 1
        y_test = np.loadtxt(os.path.join(dataset_path, "y_test.txt"), dtype=int) - 1
    except Exception as e:
        print(f"Error loading label file: {e}")
        return None, None, None, None, None
    scaler = StandardScaler()
    num_samples_train, num_channels, sequence_length = X_train.shape
    num_samples_test = X_test.shape[0]
    X_train_reshaped = X_train.transpose(1, 0, 2).reshape(num_channels, -1).T
    X_test_reshaped = X_test.transpose(1, 0, 2).reshape(num_channels, -1).T
    scaler.fit(X_train_reshaped)
    X_train_scaled_reshaped = scaler.transform(X_train_reshaped)
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
    X_train = X_train_scaled_reshaped.T.reshape(num_channels, num_samples_train, sequence_length).transpose(1, 0, 2)
    X_test = X_test_scaled_reshaped.T.reshape(num_channels, num_samples_test, sequence_length).transpose(1, 0, 2)
    ucihar_activity_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']
    return X_train, y_train, X_test, y_test, ucihar_activity_names

class UCIHAR(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1DWithRegression(nn.Module):
    def __init__(self, in_channels=9, num_classes=6, backbone_dim=256, projection_dim=256):
        super(ResNet1DWithRegression, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, backbone_dim, stride=2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim * 2, projection_dim)
        )
        self.classifier = nn.Linear(backbone_dim, num_classes)
        self.rotation_predictor = nn.Linear(backbone_dim, 3)
    def forward(self, x, return_embed=False, return_features=False, predict_rotation=False):
        features = self.feature_extractor(x).squeeze(-1)
        if return_features: return features
        if return_embed: return self.projector(features)
        if predict_rotation: return self.rotation_predictor(features)
        return self.classifier(features)

def apply_rotation_3d(x, euler_angles):
    device = x.device
    batch_size, _, seq_len = x.shape
    if isinstance(euler_angles, torch.Tensor):
        euler_angles = euler_angles.cpu().numpy()
    angles_rad = np.radians(euler_angles)
    rot_matrices = Rotation.from_euler('xyz', angles_rad).as_matrix()
    rot_matrices = torch.tensor(rot_matrices, dtype=torch.float32).to(device)
    x_reshaped = x.view(batch_size, 3, 3, seq_len)
    x_rotated = torch.einsum('bij,bgjt->bigt', rot_matrices, x_reshaped)
    return x_rotated.reshape(batch_size, 9, seq_len)

def sample_rotation_from_meridian_slice(bin_idx, batch_size, num_bins=18):
    longitude_start = -180 + (360 / num_bins) * bin_idx
    longitude_end = longitude_start + (360 / num_bins)
    angles_batch = np.zeros((batch_size, 3))
    for b in range(batch_size):
        latitude = np.random.uniform(-90, 90)
        longitude = np.random.uniform(longitude_start, longitude_end)
        lat_rad = np.radians(latitude)
        lon_rad = np.radians(longitude)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        target_direction = np.array([x, y, z])
        rotation_axis = np.cross([0, 0, 1], target_direction)
        if np.linalg.norm(rotation_axis) < 1e-10:
            if z > 0:
                rotation = Rotation.identity()
            else:
                rotation = Rotation.from_euler('x', 180, degrees=True)
        else:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(target_direction[2], -1, 1))
            rotation = Rotation.from_rotvec(rotation_axis * angle)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        angles_batch[b] = euler_angles
    return angles_batch

def barlow_twins_loss(z1, z2, lambda_param=5e-3):
    z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
    N, D = z1_norm.shape
    c = (z1_norm.T @ z2_norm) / N
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.fill_diagonal_(0).pow_(2).sum()
    return on_diag + lambda_param * off_diag

def train_supervised(model, train_loader, device, epochs=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        current_lr = optimizer.param_groups[0]['lr']
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        train_loss /= len(train_loader)
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        scheduler.step()
    training_time = time.time() - start_time
    return model, training_time

def pretrain_model(model, train_loader, device, epochs, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)
    rot_criterion = nn.MSELoss()
    num_views = 18
    bin_losses = torch.zeros(num_views)
    curriculum_start = 50
    epoch_angles_history = {1: [], 50: [], 100: [], 150: [], 200: []}
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss, total_contrast_loss, total_reg_loss = 0, 0, 0
        current_lr = optimizer.param_groups[0]['lr']
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            batch_size = x.shape[0]
            optimizer.zero_grad()
            if epoch < curriculum_start:
                selected_bins = list(range(num_views))
            else:
                selected_bins = list(range(num_views))
                difficult_bins = torch.topk(bin_losses, k=min(6, num_views)).indices.tolist()
                selected_bins.extend(difficult_bins)
            views = []
            view_angles = []
            for i in selected_bins:
                angles_xyz = sample_rotation_from_meridian_slice(i, batch_size, num_views)
                views.append(apply_rotation_3d(x, angles_xyz))
                view_angles.append(angles_xyz)
            if batch_idx == 0 and (epoch + 1) in epoch_angles_history:
                for angles_xyz in view_angles:
                    epoch_angles_history[epoch + 1].append(angles_xyz[0])
            embeddings = [model(v, return_embed=True) for v in views]
            loss_contrastive = 0
            num_pairs = 0
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    loss_contrastive += barlow_twins_loss(embeddings[i], embeddings[j])
                    num_pairs += 1
            loss_contrastive /= num_pairs
            loss = loss_contrastive
            angle_pred = model(views[0], predict_rotation=True)
            angle_target = torch.from_numpy(view_angles[0]).float().to(device) / 180.0
            loss_rotation = rot_criterion(angle_pred, angle_target)
            loss += 0.1 * loss_rotation
            loss_rotation_val = loss_rotation.item()
            bin_idx = selected_bins[0]
            bin_losses[bin_idx] = 0.9 * bin_losses[bin_idx] + 0.1 * loss_rotation_val
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_contrast_loss += loss_contrastive.item()
            total_reg_loss += loss_rotation_val
        avg_loss = total_loss / len(train_loader)
        avg_cl_loss = total_contrast_loss / len(train_loader)
        avg_reg_loss = total_reg_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}, Loss: {avg_loss:.4f}, CL Loss: {avg_cl_loss:.4f}, Reg Loss: {avg_reg_loss:.4f}")
        scheduler.step()
    pretrain_time = time.time() - start_time
    visualize_rotation_curriculum(epoch_angles_history, num_views, curriculum_start)
    return model, pretrain_time

def visualize_rotation_curriculum(epoch_angles_history, num_views, curriculum_start):
    all_epochs_data = {}
    for epoch_num in sorted(epoch_angles_history.keys()):
        if not epoch_angles_history[epoch_num]:
            continue
        accumulated_angles = []
        accumulated_colors = []
        for prev_epoch in sorted(epoch_angles_history.keys()):
            if prev_epoch <= epoch_num and epoch_angles_history[prev_epoch]:
                for angles_xyz in epoch_angles_history[prev_epoch]:
                    accumulated_angles.append(angles_xyz)
                    if prev_epoch >= curriculum_start:
                        accumulated_colors.append('blue')
                    else:
                        accumulated_colors.append('red')
        all_epochs_data[epoch_num] = (accumulated_angles, accumulated_colors)
    for epoch_num, (angles_list, colors_list) in all_epochs_data.items():
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        x_sphere = 1.0 * np.outer(np.cos(u), np.sin(v))
        y_sphere = 1.0 * np.outer(np.sin(u), np.sin(v))
        z_sphere = 1.0 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.08, color='lightgray', linewidth=0, antialiased=True, shade=False)
        num_meridians = 18
        for i in range(num_meridians):
            lon = 2 * np.pi * i / num_meridians
            lat = np.linspace(0, np.pi, 100)
            x_meridian = np.sin(lat) * np.cos(lon)
            y_meridian = np.sin(lat) * np.sin(lon)
            z_meridian = np.cos(lat)
            ax.plot(x_meridian, y_meridian, z_meridian, 'k-', alpha=0.25, linewidth=1.2)
        ax.set_axis_off()
        print(f"Epoch {epoch_num}: Plotting {len(angles_list)} points")
        for angles_xyz, color in zip(angles_list, colors_list):
            rot = Rotation.from_euler('xyz', angles_xyz, degrees=True)
            vec = rot.apply([0, 0, 1])
            ax.scatter(vec[0], vec[1], vec[2], c=color, s=15, alpha=0.6, edgecolors='none')
        ax.set_title(f'Epoch {epoch_num} - Cumulative Rotation Distribution ({len(angles_list)} views)\n' + ('Warm-up (Red)' if epoch_num < curriculum_start else 'Warm-up (Red) + Curriculum (Blue)'), fontsize=13, pad=20)
        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])
        ax.set_zlim([-1.3, 1.3])
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1,1,1])
        plt.tight_layout()
        plt.savefig(f'./rotation_viz_epoch_{epoch_num}.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

def fine_tune(model, train_loader, device, epochs=50, lr=0.0001, freeze_backbone=False):
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        print("Backbone frozen. Only classifier will be trained.")
    else:
        for param in model.parameters():
            param.requires_grad = True
        print("All parameters unfrozen. Full model will be fine-tuned.")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        current_lr = optimizer.param_groups[0]['lr']
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        train_loss /= len(train_loader)
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}, Fine-tune Loss: {train_loss:.4f}, Fine-tune Acc: {train_acc:.2f}%")
        scheduler.step()
    finetune_time = time.time() - start_time
    return model, finetune_time

def evaluate_model_standard(model, test_loader, device, activity_names):
    model.eval()
    model.to(device)
    y_true_all, y_pred_all, embeddings_list = [], [], []
    inference_time = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            start_time = time.time()
            pred = model(x)
            inference_time += time.time() - start_time
            features = model(x, return_features=True)
            embeddings_list.append(features.cpu().numpy())
            y_true_all.extend(y.cpu().numpy())
            y_pred_all.extend(pred.argmax(dim=1).cpu().numpy())
    embeddings = np.concatenate(embeddings_list)
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    acc = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    precision = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    cm = confusion_matrix(y_true_all, y_pred_all)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    flops, _ = profile(model, inputs=(torch.randn(1, 9, 128).to(device),), verbose=False)
    flops /= 1e6
    inference_time_ms = (inference_time / len(y_true_all)) * 1000
    class_f1 = f1_score(y_true_all, y_pred_all, average=None, zero_division=0)
    class_acc = []
    for i in range(len(activity_names)):
        mask = y_true_all == i
        if np.sum(mask) > 0:
            class_acc.append(accuracy_score(y_true_all[mask], y_pred_all[mask]))
        else:
            class_acc.append(0.0)
    return acc, f1, precision, recall, params, flops, inference_time_ms, cm, embeddings, y_true_all, class_acc, class_f1

def evaluate_model_rotation_fixed(model, test_loader, device, activity_names, angles):
    model.eval()
    model.to(device)
    all_results = []
    for angle in angles:
        y_true_angle, y_pred_angle = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                batch_size = len(x)
                euler_angles = np.tile([angle, angle, angle], (batch_size, 1))
                x_rot = apply_rotation_3d(x, euler_angles)
                pred = model(x_rot)
                y_true_angle.extend(y.cpu().numpy())
                y_pred_angle.extend(pred.argmax(dim=1).cpu().numpy())
        y_true_angle = np.array(y_true_angle)
        y_pred_angle = np.array(y_pred_angle)
        acc = accuracy_score(y_true_angle, y_pred_angle)
        f1 = f1_score(y_true_angle, y_pred_angle, average='macro', zero_division=0)
        class_f1 = f1_score(y_true_angle, y_pred_angle, average=None, zero_division=0)
        class_acc = []
        for i in range(len(activity_names)):
            mask = y_true_angle == i
            if np.sum(mask) > 0:
                class_acc.append(accuracy_score(y_true_angle[mask], y_pred_angle[mask]))
            else:
                class_acc.append(0.0)
        result = {'Angle': angle, 'Overall': f'{acc:.4f}/{f1:.4f}'}
        for i, act in enumerate(activity_names):
            result[act] = f'{class_acc[i]:.4f}/{class_f1[i]:.4f}'
        all_results.append(result)
    return pd.DataFrame(all_results)

def evaluate_model_rotation_random(model, test_loader, device, activity_names):
    model.eval()
    model.to(device)
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            batch_size = len(x)
            euler_angles = np.random.uniform(-180, 180, (batch_size, 3))
            x_rot = apply_rotation_3d(x, euler_angles)
            pred = model(x_rot)
            y_true_all.extend(y.cpu().numpy())
            y_pred_all.extend(pred.argmax(dim=1).cpu().numpy())
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    acc = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    precision = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    class_f1 = f1_score(y_true_all, y_pred_all, average=None, zero_division=0)
    class_precision = precision_score(y_true_all, y_pred_all, average=None, zero_division=0)
    class_recall = recall_score(y_true_all, y_pred_all, average=None, zero_division=0)
    class_acc = []
    for i in range(len(activity_names)):
        mask = y_true_all == i
        if np.sum(mask) > 0:
            class_acc.append(accuracy_score(y_true_all[mask], y_pred_all[mask]))
        else:
            class_acc.append(0.0)
    return acc, f1, precision, recall, class_acc, class_f1, class_precision, class_recall

def plot_tsne(features, labels, activity_names, dataset_name="Dataset", samples_per_class=300):
    sampled_features, sampled_labels = [], []
    for i in range(len(activity_names)):
        class_mask = labels == i
        class_features = features[class_mask]
        if len(class_features) > samples_per_class:
            idx = np.random.choice(len(class_features), samples_per_class, replace=False)
            sampled_features.append(class_features[idx])
            sampled_labels.append(labels[class_mask][idx])
        else:
            sampled_features.append(class_features)
            sampled_labels.append(labels[class_mask])
    features = np.vstack(sampled_features)
    labels = np.hstack(sampled_labels)
    features_2d = TSNE(n_components=2, perplexity=20, learning_rate=3000).fit_transform(features)
    colors = ["#FF0000", "#FFA500", "#FFFF00", "#008000", "#00FFFF", "#0000FF", "#800080", "#FFC0CB", "#A52A2A", "#000000", "#808080", "#FFD700"]
    plt.figure(figsize=(8, 6))
    for i, activity in enumerate(activity_names):
        mask = labels == i
        if np.any(mask):
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], color=colors[i], marker='o', s=20, alpha=0.35)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=activity, markerfacecolor=colors[i], markersize=7) for i, activity in enumerate(activity_names)]
    plt.legend(handles=handles, title="Activities", fontsize=9, framealpha=1)
    plt.xlabel("t-SNE Component 1", fontsize=13)
    plt.ylabel("t-SNE Component 2", fontsize=13)
    plt.grid(False)
    plt.savefig(f'./{dataset_name}_tsne.png', dpi=500)
    plt.show()

def plot_confusion_matrix(cm, activity_names, title, save_path):
    labels = ['Walking', 'Walking\nUpstairs', 'Walking\nDownstairs', 'Sitting', 'Standing', 'Laying']
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df = pd.DataFrame(cm_norm, index=labels, columns=labels)
    annot = df.copy().astype(str)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.iloc[i, j]
            annot.iloc[i, j] = f"{v:.2f}"
    plt.figure(figsize=(6, 5))
    sns.heatmap(df, annot=annot.values, fmt="", cmap="Blues", cbar=True, annot_kws={"size": 14}, vmin=0, vmax=1)
    plt.xticks(rotation=90, fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor('black')
    plt.xlabel('Predicted Label', fontsize=10)
    plt.ylabel('True Label', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dataset_path = '/workspace'
    X_train, y_train, X_test, y_test, activity_names = load_uci_har_raw(dataset_path)
    if X_train is None:
        print("Failed to load data.")
        return
    train_dataset = UCIHAR(X_train, y_train)
    test_dataset = UCIHAR(X_test, y_test)
    BATCH_SIZE = 512
    train_loader_full = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    all_supervised_results = []
    all_rotation_results = []
    print("\n" + "="*50)
    print("1) SUPERVISED LEARNING (100 epochs)")
    print("="*50)
    model_sup = ResNet1DWithRegression().to(device)
    model_sup, train_time_sup = train_supervised(model_sup, train_loader_full, device, epochs=100, lr=0.001)
    acc, f1, prec, rec, params, flops, inf_time, cm, emb, y_true, class_acc, class_f1 = evaluate_model_standard(model_sup, test_loader, device, activity_names)
    print("\n--- Supervised: Overall Metrics ---")
    overall_df = pd.DataFrame({'Metric': ['Accuracy', 'F1', 'Precision', 'Recall', 'Params (M)', 'FLOPs (M)', 'Inference Time (ms)', 'Training Time (s)'], 'Value': [f'{acc:.4f}', f'{f1:.4f}', f'{prec:.4f}', f'{rec:.4f}', f'{params:.4f}', f'{flops:.4f}', f'{inf_time:.4f}', f'{train_time_sup:.2f}']})
    display(overall_df)
    print("\n--- Supervised: Classwise Metrics ---")
    classwise_df = pd.DataFrame({'Activity': activity_names, 'Accuracy': [f'{a:.4f}' for a in class_acc], 'F1': [f'{f:.4f}' for f in class_f1]})
    display(classwise_df)
    plot_confusion_matrix(cm, activity_names, 'Supervised', './supervised_confusion.png')
    plot_tsne(emb, y_true, activity_names, 'supervised')
    print("\n--- Supervised: Rotation Test (Fixed Angles, 20deg intervals) ---")
    angles_fixed = np.arange(-180, 181, 20)
    rot_df_sup_fixed = evaluate_model_rotation_fixed(model_sup, test_loader, device, activity_names, angles_fixed)
    rot_df_sup_fixed_display = rot_df_sup_fixed.set_index('Angle')
    display(rot_df_sup_fixed_display)
    avg_overall_acc_fixed = []
    avg_overall_f1_fixed = []
    for val in rot_df_sup_fixed['Overall']:
        parts = val.split('/')
        avg_overall_acc_fixed.append(float(parts[0]))
        avg_overall_f1_fixed.append(float(parts[1]))
    print("\n--- Supervised: Overall Rotation Performance (Fixed) ---")
    overall_rot_summary_fixed = pd.DataFrame({'Metric': ['Average Rotation Accuracy', 'Average Rotation F1'], 'Value': [f'{np.mean(avg_overall_acc_fixed):.4f}', f'{np.mean(avg_overall_f1_fixed):.4f}']})
    display(overall_rot_summary_fixed)
    avg_rot_class_fixed = []
    for act in activity_names:
        acc_vals = []
        f1_vals = []
        for val in rot_df_sup_fixed[act]:
            parts = val.split('/')
            acc_vals.append(float(parts[0]))
            f1_vals.append(float(parts[1]))
        avg_rot_class_fixed.append({'Activity': act, 'Avg_Rot_Acc': f'{np.mean(acc_vals):.4f}', 'Avg_Rot_F1': f'{np.mean(f1_vals):.4f}'})
    avg_rot_class_df_fixed = pd.DataFrame(avg_rot_class_fixed)
    print("\n--- Supervised: Classwise Rotation Performance (Fixed) ---")
    display(avg_rot_class_df_fixed)
    print("\n--- Supervised: Rotation Test (Fully Random Angles) ---")
    print("Starting random rotation evaluation...")
    acc_rand, f1_rand, prec_rand, rec_rand, class_acc_rand, class_f1_rand, class_prec_rand, class_rec_rand = evaluate_model_rotation_random(model_sup, test_loader, device, activity_names)
    print(f"Random rotation evaluation completed. Acc: {acc_rand:.4f}, F1: {f1_rand:.4f}")
    print("\n--- Supervised: Overall Rotation Performance (Random) ---")
    overall_rot_summary_rand = pd.DataFrame({'Metric': ['Rotation Accuracy', 'Rotation F1', 'Rotation Precision', 'Rotation Recall'], 'Value': [f'{acc_rand:.4f}', f'{f1_rand:.4f}', f'{prec_rand:.4f}', f'{rec_rand:.4f}']})
    display(overall_rot_summary_rand)
    print("\n--- Supervised: Classwise Rotation Performance (Random) ---")
    classwise_rot_rand_df = pd.DataFrame({'Activity': activity_names, 'Accuracy': [f'{a:.4f}' for a in class_acc_rand], 'F1': [f'{f:.4f}' for f in class_f1_rand]})
    display(classwise_rot_rand_df)
    all_supervised_results.append({'Model': 'Supervised', 'Acc': acc, 'F1': f1, 'Precision': prec, 'Recall': rec, 'Params': params, 'FLOPs': flops, 'Inf_Time': inf_time, 'Train_Time': train_time_sup})
    all_rotation_results.append({'Model': 'Supervised', 'Avg_Rot_Acc_Fixed': f'{np.mean(avg_overall_acc_fixed):.4f}', 'Avg_Rot_F1_Fixed': f'{np.mean(avg_overall_f1_fixed):.4f}', 'Rot_Acc_Random': f'{acc_rand:.4f}', 'Rot_F1_Random': f'{f1_rand:.4f}', 'Rot_Prec_Random': f'{prec_rand:.4f}', 'Rot_Rec_Random': f'{rec_rand:.4f}'})
    torch.save(model_sup.state_dict(), './model_supervised.pth')
    data_percentages = [100, 50, 10, 5, 1]
    for pct in data_percentages:
        print("\n" + "="*50)
        print(f"2) SSL (Contrastive + Curriculum + Residual) with {pct}% data")
        print("="*50)
        num_samples = int(len(train_dataset) * pct / 100)
        indices = np.random.choice(len(train_dataset), num_samples, replace=False)
        train_subset = Subset(train_dataset, indices)
        train_loader_subset = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        pretrain_lr = 0.001
        finetune_lr = 0.0001
        model_ssl = ResNet1DWithRegression().to(device)
        model_ssl, pretrain_time = pretrain_model(model_ssl, train_loader_subset, device, epochs=200, lr=pretrain_lr)
        final_model = ResNet1DWithRegression().to(device)
        pretrained_dict = {k: v for k, v in model_ssl.state_dict().items() if "rotation_predictor" not in k and "projector" not in k}
        final_model.load_state_dict(pretrained_dict, strict=False)
        final_model, finetune_time = fine_tune(final_model, train_loader_subset, device, epochs=50, lr=finetune_lr)
        acc, f1, prec, rec, params, flops, inf_time, cm, emb, y_true, class_acc, class_f1 = evaluate_model_standard(final_model, test_loader, device, activity_names)
        print(f"\n--- SSL {pct}%: Overall Metrics ---")
        overall_df = pd.DataFrame({'Metric': ['Accuracy', 'F1', 'Precision', 'Recall', 'Params (M)', 'FLOPs (M)', 'Inference Time (ms)', 'Pretrain Time (s)', 'Finetune Time (s)'], 'Value': [f'{acc:.4f}', f'{f1:.4f}', f'{prec:.4f}', f'{rec:.4f}', f'{params:.4f}', f'{flops:.4f}', f'{inf_time:.4f}', f'{pretrain_time:.2f}', f'{finetune_time:.2f}']})
        display(overall_df)
        print(f"\n--- SSL {pct}%: Classwise Metrics ---")
        classwise_df = pd.DataFrame({'Activity': activity_names, 'Accuracy': [f'{a:.4f}' for a in class_acc], 'F1': [f'{f:.4f}' for f in class_f1]})
        display(classwise_df)
        print(f"\n--- SSL {pct}%: Rotation Test (Fixed Angles, 20deg intervals) ---")
        rot_df_fixed = evaluate_model_rotation_fixed(final_model, test_loader, device, activity_names, angles_fixed)
        rot_df_fixed_display = rot_df_fixed.set_index('Angle')
        display(rot_df_fixed_display)
        avg_overall_acc_fixed = []
        avg_overall_f1_fixed = []
        for val in rot_df_fixed['Overall']:
            parts = val.split('/')
            avg_overall_acc_fixed.append(float(parts[0]))
            avg_overall_f1_fixed.append(float(parts[1]))
        print(f"\n--- SSL {pct}%: Overall Rotation Performance (Fixed) ---")
        overall_rot_summary_fixed = pd.DataFrame({'Metric': ['Average Rotation Accuracy', 'Average Rotation F1'], 'Value': [f'{np.mean(avg_overall_acc_fixed):.4f}', f'{np.mean(avg_overall_f1_fixed):.4f}']})
        display(overall_rot_summary_fixed)
        avg_rot_class_fixed = []
        for act in activity_names:
            acc_vals = []
            f1_vals = []
            for val in rot_df_fixed[act]:
                parts = val.split('/')
                acc_vals.append(float(parts[0]))
                f1_vals.append(float(parts[1]))
            avg_rot_class_fixed.append({'Activity': act, 'Avg_Rot_Acc': f'{np.mean(acc_vals):.4f}', 'Avg_Rot_F1': f'{np.mean(f1_vals):.4f}'})
        avg_rot_class_df_fixed = pd.DataFrame(avg_rot_class_fixed)
        print(f"\n--- SSL {pct}%: Classwise Rotation Performance (Fixed) ---")
        display(avg_rot_class_df_fixed)
        print(f"\n--- SSL {pct}%: Rotation Test (Fully Random Angles) ---")
        acc_rand, f1_rand, prec_rand, rec_rand, class_acc_rand, class_f1_rand, class_prec_rand, class_rec_rand = evaluate_model_rotation_random(final_model, test_loader, device, activity_names)
        print(f"\n--- SSL {pct}%: Overall Rotation Performance (Random) ---")
        overall_rot_summary_rand = pd.DataFrame({'Metric': ['Rotation Accuracy', 'Rotation F1', 'Rotation Precision', 'Rotation Recall'], 'Value': [f'{acc_rand:.4f}', f'{f1_rand:.4f}', f'{prec_rand:.4f}', f'{rec_rand:.4f}']})
        display(overall_rot_summary_rand)
        print(f"\n--- SSL {pct}%: Classwise Rotation Performance (Random) ---")
        classwise_rot_rand_df = pd.DataFrame({'Activity': activity_names, 'Accuracy': [f'{a:.4f}' for a in class_acc_rand], 'F1': [f'{f:.4f}' for f in class_f1_rand]})
        display(classwise_rot_rand_df)
        all_supervised_results.append({'Model': f'SSL_{pct}%', 'Acc': acc, 'F1': f1, 'Precision': prec, 'Recall': rec, 'Params': params, 'FLOPs': flops, 'Inf_Time': inf_time, 'Pretrain_Time': pretrain_time, 'Finetune_Time': finetune_time})
        all_rotation_results.append({'Model': f'SSL_{pct}%', 'Avg_Rot_Acc_Fixed': f'{np.mean(avg_overall_acc_fixed):.4f}', 'Avg_Rot_F1_Fixed': f'{np.mean(avg_overall_f1_fixed):.4f}', 'Rot_Acc_Random': f'{acc_rand:.4f}', 'Rot_F1_Random': f'{f1_rand:.4f}', 'Rot_Prec_Random': f'{prec_rand:.4f}', 'Rot_Rec_Random': f'{rec_rand:.4f}'})
        if pct == 100:
            plot_confusion_matrix(cm, activity_names, f'SSL_{pct}%', f'./ssl_{pct}_confusion.png')
            plot_tsne(emb, y_true, activity_names, f'ssl_{pct}')
            torch.save(final_model.state_dict(), f'./model_ssl_{pct}.pth')
        else:
            torch.save(final_model.state_dict(), f'./model_ssl_{pct}.pth')
    print("\n" + "="*60)
    print("3) FINAL SUMMARY: SUPERVISED PERFORMANCE")
    print("="*60)
    summary_sup_df = pd.DataFrame(all_supervised_results)
    summary_sup_df = summary_sup_df[['Model', 'Acc', 'F1', 'Precision', 'Recall', 'Params', 'FLOPs', 'Inf_Time', 'Train_Time', 'Pretrain_Time', 'Finetune_Time']]
    summary_sup_df = summary_sup_df.fillna('-')
    display(summary_sup_df)
    print("\n" + "="*60)
    print("4) FINAL SUMMARY: ROTATION PERFORMANCE")
    print("="*60)
    summary_rot_df = pd.DataFrame(all_rotation_results)
    display(summary_rot_df)

if __name__ == '__main__':
    main()