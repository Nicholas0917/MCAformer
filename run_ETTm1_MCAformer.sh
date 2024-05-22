if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
  mkdir ./logs/LongForecasting
fi
gpu=0
station_type=adaptive
model_name=MCAformer

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/ETT-small \
  --data_path ETTm1.csv \
  --model_id ETTm1_168_168 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 168 \
  --label_len 0 \
  --pred_len 168 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 1 \
  --n_heads 8 \
  --d_model 32 \
  --d_ff 2048 \
  --batch_size 64 \
  --dropout 0.21738303694726563 \
  --period_len 12 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/ETT-small \
  --data_path ETTm1.csv \
  --model_id ETTm1_168_336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 168 \
  --label_len 0 \
  --pred_len 336 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 2 \
  --n_heads 16 \
  --d_model 32 \
  --d_ff 1024 \
  --batch_size 64 \
  --dropout 0.13185501712488573 \
  --period_len 12 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/ETT-small \
  --data_path ETTm1.csv \
  --model_id ETTm1_336_336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 336 \
  --label_len 0 \
  --pred_len 336 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 2 \
  --n_heads 8 \
  --d_model 32 \
  --d_ff 1024 \
  --batch_size 64 \
  --dropout 0.6521168569965896 \
  --period_len 12 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/ETT-small \
  --data_path ETTm1.csv \
  --model_id ETTm1_336_720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 336 \
  --label_len 0 \
  --pred_len 720 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 1 \
  --n_heads 16 \
  --d_model 32 \
  --d_ff 2048 \
  --batch_size 64 \
  --dropout 0.5893813568379566 \
  --period_len 12 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/ETT-small \
  --data_path ETTm1.csv \
  --model_id ETTm1_720_720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 720 \
  --label_len 0 \
  --pred_len 720 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 1 \
  --n_heads 4 \
  --d_model 128 \
  --d_ff 2048 \
  --batch_size 64 \
  --dropout 0.451646645908165 \
  --period_len 12 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/ETT-small \
  --data_path ETTm1.csv \
  --model_id ETTm1_720_1440 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 720 \
  --label_len 0 \
  --pred_len 1440 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 1 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 1024 \
  --batch_size 64 \
  --dropout 0.6633 \
  --period_len 12 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/ETT-small \
  --data_path ETTm1.csv \
  --model_id ETTm1_1440_1440 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 1440 \
  --label_len 0 \
  --pred_len 1440 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 2 \
  --n_heads 8 \
  --d_model 32 \
  --d_ff 256 \
  --batch_size 64 \
  --dropout 0.4035055463448366 \
  --period_len 12 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1