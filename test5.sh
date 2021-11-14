
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=7 python experiments/bert/run_interval_classifier.py \
  --task_name interval \
  --do_eval \
  --do_train \
  --do_lower_case \
  --data_dir evaluator/test5 \
  --test \
  --bert_model bert-base-uncased \
  --max_seq_length 32 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --expname 1114 \
  --output_dir ./test_fold5_2 \
  --ckpt ./experiments/bert/log/baseline_final.pth \
  --sutime_jars_path ./jars

python3 ./eval_5_fold_dataset/eval_tcr.py --test_file ./evaluator/test5/test.tsv --prediction_file ./test_fold5_2/eval_outputs.txt 
