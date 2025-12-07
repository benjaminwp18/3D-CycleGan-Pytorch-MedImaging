import sys
sys.path.insert(0, '..')  # Insert so parent dir modules take priority

import argparse
import json
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from typing import Optional

from utils.oasis1_losses import make_focal_loss
from models.oasis1_vgg16_2d import VGG16OASIS2D
from utils.oasis1_metrics import compute_all_metrics


# ImageNet normalization for VGG16
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_val_test_transform():
    """Common transform for val and test (all regimes)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

class OASIS12DSliceDataset(Dataset):
    """Dataset for 2D slices from OASIS-1 with regime support."""

    def __init__(self, slices_dir: Path = Path('results') / 'slices', transform: Optional[transforms.Compose] = None):
        """
        Args:
            transform: Optional transform (if None, uses regime-specific transform)
        """

        # Collect all slice files
        self.samples = []

        # Always load real slices
        for label_dir in slices_dir.iterdir():
            if not label_dir.is_dir():
                continue

            original_label = int(label_dir.name)
            # Convert to binary: 0 = Healthy, 1 = AD (Mild+Moderate only)
            # Drop MCI (label 1) samples
            if original_label == 1:  # Skip MCI samples
                continue
            binary_label = 0 if original_label == 0 else 1  # 0=Healthy, 1=AD (label 2)

            for slice_file in label_dir.glob("*.npy"):
                subject_id = slice_file.stem.rsplit('_', 1)[0]
                self.samples.append((str(slice_file), binary_label, subject_id))

        # Set transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_val_test_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        slice_path, label, subject_id = self.samples[idx]

        # Load slice: (3, H, W) as np.float32
        slice_data = np.load(slice_path)

        # Convert to PIL-compatible format (H, W, C)
        if slice_data.shape[0] == 3:
            slice_data = np.transpose(slice_data, (1, 2, 0))

        # Normalize slice to [0, 1] range first, then scale to [0, 255] for PIL
        slice_min, slice_max = slice_data.min(), slice_data.max()
        if slice_max > slice_min:
            slice_data = (slice_data - slice_min) / (slice_max - slice_min + 1e-8)
        slice_data = (slice_data * 255).astype(np.uint8)

        # Apply transforms
        slice_data = self.transform(slice_data)

        return slice_data, label, subject_id


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    metrics = compute_all_metrics(
        np.array(all_labels),
        np.array(all_preds),
        y_probs=np.array(all_probs),
        num_classes=2
    )
    return epoch_loss, metrics

def compute_class_counts(dataset):
    """Compute class counts from dataset."""
    class_counts = {}
    for _, label, _ in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts

# CHECKPOINT_PATH = Path('..') / 'checkpoints' / 'oasis1_vgg16_2d' / 'no_aug' / 'best.pth'
CHECKPOINT_PATH = Path('..') / 'checkpoints' / 'oasis1_vgg16_2d' / 'classical' / 'best.pth'
BATCH_SIZE_PHASE2 = 64

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--slices', type=str, default='./results/slices/test', help='Path to dir where label folders (0, 1) containing slices will be created')
    parser.add_argument('--json', type=str, default='test', help='Prefix for name of json file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    slices_path = Path(args.slices)
    test_dataset = OASIS12DSliceDataset(slices_dir=slices_path)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_PHASE2, shuffle=False, num_workers=4)

    model = VGG16OASIS2D(num_classes=2).to(device)

    class_counts = compute_class_counts(test_dataset)
    criterion = make_focal_loss(class_counts, device=device, gamma=2.0)

    print('\nEvaluating on test set...')
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Metrics: {json.dumps(test_metrics, indent=2)}')

    # Save test metrics
    output_path = Path('results') / 'metrics' / f'{args.json}_metrics.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f'\nTest metrics saved to {output_path}')