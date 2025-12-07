# Preprocess
 1. mv_files.py
 2. preprocess.py

# Train
train.sh (train.py w/args)

# Predict
predict.sh (test.py w/args)

# Evaluate
slice_results.sh
eval.sh

# Notes
lambda_A/lambda_B: cycle loss (def 10)
lambda_co_A/lambda_co_B: adv loss (def 2)
lambda_identity: idt loss (0.5 * cycle loss weight)
