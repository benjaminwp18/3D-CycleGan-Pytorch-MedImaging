from pathlib import Path
import csv
import json
import shutil

#CSV_PATH = Path('..') / 'oasis' / 'oasis_cross-sectional_demographics.csv'
LABELS_PATH = Path('..') / 'metadata' / 'oasis1_labels.csv'
SPLITS_PATH = Path('..') / 'metadata' / 'oasis1_splits.json'
OASIS_PATH = Path('..') / 'oasis'
DATASET_PATH = Path('Data_folder')
DISC_PATHS = list([OASIS_PATH / f'disc{i}' for i in range(1, 12 + 1)])

def get_scan_dir_path(subject_id: str) -> Path:
    for disc_path in DISC_PATHS:
        for scan_dir_path in disc_path.iterdir():
            if scan_dir_path.stem == subject_id:
                return scan_dir_path
    raise FileNotFoundError(subject_id)

SPLITS: dict[str, list[str]] = {'train': [], 'test': []}
with open(SPLITS_PATH) as f:
    splits_dict = json.load(f)
    SPLITS['train'] = splits_dict['train'] + splits_dict['val']
    SPLITS['test'] = splits_dict['test']

with open(LABELS_PATH, mode ='r') as csv_file:
    csv_dict = csv.DictReader(csv_file)
    for lines in csv_dict:
        label = int(lines['label'])
        if label in (0, 2):
            scan_dir_path = get_scan_dir_path(lines['subject_id'])
            split = 'train' if scan_dir_path.stem in SPLITS['train'] else 'test'
            domain = 'images' if label == 0 else 'labels'
            dest_path = DATASET_PATH / split / domain
            print(scan_dir_path, '->', DATASET_PATH / split / domain / scan_dir_path.stem)

            src_path = scan_dir_path / 'PROCESSED' / 'MPRAGE' / 'T88_111'
            hdr_and_img = list(src_path.glob('*_t88_gfc.*'))
            if not src_path.exists():
                raise FileNotFoundError(src_path)
            if (len(hdr_and_img) != 2):
                raise FileNotFoundError(f'Should have found exactly 2 files (hdr & img), instead found {hdr_and_img}')
            for filepath in hdr_and_img:
                shutil.copy(filepath, dest_path / filepath.name)
