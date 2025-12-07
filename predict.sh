# python test.py --gpu_ids 0 --batch_size 1 --workers 6 --model_suffix _A \
#     --image Data_folder/test/images/OAS1_0033_MR1_mpr_n4_anon_111_t88_gfc.nii \
#     --result results/OAS1_0033_MR1_mpr_n4_anon_111_t88_gfc_fake.nii



# python test.py --name deep_disc --n_layers_D 4 \
#     --gpu_ids 0 --batch_size 1 --workers 6 --model_suffix _A \
#     --image_dir Data_folder/test/images \
#     --result_dir results/deep_disc/fake_labels/test

# python test.py --name deep_disc --n_layers_D 4 \
#     --gpu_ids 0 --batch_size 1 --workers 6 --model_suffix _B \
#     --image_dir Data_folder/test/labels \
#     --result_dir results/deep_disc/fake_images/test

# python test.py --name deep_disc --n_layers_D 4 \
#     --gpu_ids 0 --batch_size 1 --workers 6 --model_suffix _A \
#     --image_dir Data_folder/train/images \
#     --result_dir results/deep_disc/fake_labels/train

# python test.py --name deep_disc --n_layers_D 4 \
#     --gpu_ids 0 --batch_size 1 --workers 6 --model_suffix _B \
#     --image_dir Data_folder/train/labels \
#     --result_dir results/deep_disc/fake_images/train



python test.py --name deep_disc --n_layers_D 4 \
    --gpu_ids 0 --batch_size 1 --workers 6 --model_suffix _A \
    --image_dir Data_folder/splits/1_3 \
    --result_dir results/splits/1_3/

python test.py --name deep_disc --n_layers_D 4 \
    --gpu_ids 0 --batch_size 1 --workers 6 --model_suffix _A \
    --image_dir Data_folder/splits/1_1_5 \
    --result_dir results/splits/1_1_5/
