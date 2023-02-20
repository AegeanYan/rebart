#!/bin/bash


DATA_DIR="data/amazon_full_5"
OUT_DIR="outputs/reorder_exp/bart_large-amazon_full_5"
# NEW_DIR="outputs/reorder_exp/R_amazon_full"
# mkdir -p ${NEW_DIR}
# cp $0 ${NEW_DIR}
mkdir -p ${OUT_DIR}
cp $0 ${OUT_DIR}

python -m source.encoder_decoder \
    --train_file ${DATA_DIR}/train.jsonl \
    --eval_data_file ${DATA_DIR}/dev.jsonl \
    --out_dir $OUT_DIR \
    --model_type facebook/bart-large \
    --model_name_or_path facebook/bart-large \
    --device 7 \
    --do_train \
    --do_eval \
    --save_total_limit 1 \
    --num_train_epochs 1 \
    --logging_steps 3000 \
    --gradient_accumulation_steps 8 \
    --train_batch_size 128 \
    --eval_batch_size 8 \
    --overwrite_out_dir \
    --max_input_length 1024 \
    --max_output_length 40 \
    --task index_with_sep \
    --continue_training \
    # --continue_new_out_dir $NEW_DIR \
    $@
#--overwrite_cache \