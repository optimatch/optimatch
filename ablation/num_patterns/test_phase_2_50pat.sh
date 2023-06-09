python run.py \
    --num_clusters 50 \
    --model_name=phase_two_model_50pat.bin \
    --output_dir=./saved_models \
    --do_test \
    --codebook_initialized \
    --train_data_file=../../data/processed_train.csv \
    --eval_data_file=../../data/processed_val.csv \
    --test_data_file=../../data/processed_test.csv \
    --encoder_block_size 512 \
    --max_num_statements 155 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --seed 123456  2>&1 | tee test_phase_2_50pat.log