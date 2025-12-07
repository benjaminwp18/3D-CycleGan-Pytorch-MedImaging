import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from scipy.ndimage import zoom

METADATA_LABELS_CSV = Path('..') / 'metadata' / 'oasis1_labels.csv'

def normalize_volume(volume: np.ndarray, clip_percentiles: tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    """Normalize volume: clip to percentiles, then z-score."""
    # Clip to percentiles
    p_low, p_high = np.percentile(volume, clip_percentiles)
    volume = np.clip(volume, p_low, p_high)

    # Z-score normalization
    mean = volume.mean()
    std = volume.std()
    if std > 0:
        volume = (volume - mean) / std
    else:
        volume = volume - mean

    return volume

def prepare_2d_slices(src_path: Path = Path('results') / 'fake_labels',
                      dst_path: Path = Path('results') / 'slices'):
    """Prepare 2D slices from OASIS-1 volumes.

    Extracts slices at indices 60-80 and 100-120 from axial view.
    Normalizes per volume and saves as 224x224 3-channel images.
    """

    print(f'Slicing from {src_path} to {dst_path}')

    # Load metadata
    if not METADATA_LABELS_CSV.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_LABELS_CSV}")

    labels_df = pd.read_csv(METADATA_LABELS_CSV)

    # Process each subject
    label_map = labels_df.set_index('subject_id')['label'].to_dict()

    for image_path in src_path.glob('*.nii'):
        subject_id = '_'.join(image_path.stem.split('_')[:3])

        label = label_map[subject_id]
        # Swap labels; assuming GAN has changed each image's domain
        if label == 0:
            label = 2
        elif label == 2:
            label = 0
        else:
            raise Exception('Illegal mild disease label')
        split_dir = dst_path / str(label)
        split_dir.mkdir(parents=True, exist_ok=True)

        existing = list(split_dir.glob(f"{subject_id}_*.npy"))
        if existing:
            continue

        try:
            # Load volume
            img = nib.load(str(image_path))
            volume = np.array(img.dataobj)

            # Check volume dimensions
            if len(volume.shape) != 3:
                print(f"Skipping {subject_id}: volume has {len(volume.shape)} dimensions, expected 3. Shape: {volume.shape}")
                continue

            # Validate volume has reasonable dimensions
            if any(dim < 10 for dim in volume.shape):
                print(f"Skipping {subject_id}: volume has suspiciously small dimensions: {volume.shape}")
                continue

            volume = normalize_volume(volume)

            # Extract axial slices (assuming volume is in RAS+ orientation)
            # Volume shape should be (H, W, D) where D is the slice dimension
            # Extract slices at indices 60-80 and 100-120
            slice_indices = list(range(60, 81)) + list(range(100, 121))

            # Ensure we don't exceed volume bounds
            max_slice = min(volume.shape[2], max(slice_indices) + 1)
            slice_indices = [idx for idx in slice_indices if idx < max_slice]

            if not slice_indices:
                print(f"Skipping {subject_id}: no valid slice indices for volume shape {volume.shape}")
                continue

            for slice_idx in slice_indices:
                slice_2d = volume[:, :, slice_idx]

                # Skip if slice is empty or has wrong dimensions
                if slice_2d.size == 0 or len(slice_2d.shape) != 2:
                    continue

                # Convert to 3 channels (repeat)
                slice_3ch = np.stack([slice_2d] * 3, axis=0)

                # Resize to 224x224
                if slice_3ch.shape[1] != 224 or slice_3ch.shape[2] != 224:
                    if slice_3ch.shape[1] > 0 and slice_3ch.shape[2] > 0:
                        zoom_factors = (1.0, 224 / slice_3ch.shape[1], 224 / slice_3ch.shape[2])
                        slice_3ch = zoom(slice_3ch, zoom_factors, order=1)
                    else:
                        continue

                # Save
                save_path = split_dir / f"{subject_id}_{slice_idx}.npy"
                np.save(save_path, slice_3ch.astype(np.float32))

            print(f"Processed {subject_id}")
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--src', type=str, default='./results/fake_labels/test', help='Path to dir containing generated 3D nii images')
    parser.add_argument('--dst', type=str, default='./results/slices/test', help='Path to dir where label folders (0, 1) containing slices will be created')

    args = parser.parse_args()

    prepare_2d_slices(Path(args.src), Path(args.dst))