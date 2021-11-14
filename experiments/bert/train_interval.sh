#!/usr/bin/env bash

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=8 python run_interval_classifier.py \
  --task_name interval \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ../../dataset \
  --bert_model bert-base-uncased \
  --max_seq_length 32 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30 \
  --expname 1113 \
  --output_dir ./1113_run \
  --ckpt ./baseline.pth \
  --sutime_jars_path ../../jars


  # python run_interval_classifier.py \
  # --task_name interval \
  # --do_train \
  # --do_eval \
  # --do_lower_case \
  # --data_dir ../../dataset \
  # --bert_model bert-base-uncased \
  # --max_seq_length 32 \
  # --train_batch_size 32 \
  # --learning_rate 2e-5 \
  # --num_train_epochs 30.0 \
  # --expname 1113 \
  # --output_dir ./1113_run \
  # --sutime_jars_path ../../jars


  