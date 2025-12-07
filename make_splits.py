from pathlib import Path
import json
import shutil

# import predict

json_path_1_1_5 = Path('..') / 'metadata' / 'oasis1_splits_1_1_5_for_partner.json'
json_path_1_3 = Path('..') / 'metadata' / 'oasis1_splits_1_3_for_partner.json'
src_path_train = Path('Data_folder') / 'train' / 'images'
src_path_test = Path('Data_folder') / 'test' / 'images'
dst_path = Path('Data_folder') / 'splits'

PREDICT_OPTS = {
    'name': 'deep_disc',
    'n_layers_D': 4,
    'gpu_ids': 0,
    'batch_size': 1,
    'workers': 6,
    'model_suffix': '_A',
    'image_dir': 'Data_folder/test/images',
    'result_dir': 'results/deep_disc/fake_labels/test',
}

for split, split_path in (('1_1_5', json_path_1_1_5), ('1_3', json_path_1_3)):
    with open(split_path) as f:
        split_json = json.load(f)

    healthy = split_json['splits']['train']['healthy_subjects']

    for subject in healthy:
        print(f'Finding {subject}')
        img_paths = list(src_path_train.glob(subject + '*.nii'))
        if len(img_paths) == 0:
            img_paths = list(src_path_test.glob(subject + '*.nii'))
        if len(img_paths) != 1:
            raise Exception(f'Found {len(img_paths)} matching images instead of 1: {img_paths}')
        full_dst = dst_path / split
        full_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_paths[0], full_dst)

        # predict.main(img_paths[0])