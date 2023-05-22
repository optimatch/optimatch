python velvet_main.py \
    --model_name=velvet_model.bin \
    --output_dir=./saved_models \
    --do_train \
    --do_test \
    --train_data_file=../LineVD/linevd_data/processed_train.csv \
    --eval_data_file=../LineVD/linevd_data/processed_val.csv \
    --test_data_file=../LineVD/linevd_data/processed_test.csv \
    --epochs 10 \
    --encoder_block_size 512 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train_velvet.log