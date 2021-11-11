#!/usr/bin/env bash

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=7 python experiments/bert/run_interval_classifier.py \
  --task_name temporal \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir dataset \
  --bert_model bert-base-uncased \
  --interval_model  
  --interval_backbone bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --expname 1111_interval \
  --output_dir ./bert_output \
