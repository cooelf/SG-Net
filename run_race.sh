python run_race.py \
  --bert_model bert-large-cased-whole-word-masking \
  --do_train \
  --do_eval \
  --train_batch_size 8 \
  --eval_batch_size 20 \
  --num_train_epochs 3.0 \
  --max_seq_length 384 \
  --learning_rate 8e-6 \
  --output_dir output_path
