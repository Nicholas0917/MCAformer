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
  --data_path ETTm2.csv \
  --model_id ETTm2_168_168 \
  --model $model_name \
  --data ETTm2 \
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
  --e_layers 2 \
  --n_heads 8 \
  --d_model 32 \
  --d_ff 1024 \
  --batch_size 64 \
  --dropout 0.3763 \
  --period_len 24 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/ETT-small \
  --data_path ETTm2.csv \
  --model_id ETTm2_168_336 \
  --model $model_name \
  --data ETTm2 \
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
  --e_layers 1 \
  --n_heads 4 \
  --d_model 128 \
  --d_ff 256 \
  --batch_size 64 \
  --dropout 0.4343 \
  --period_len 12 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/ETT-small \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_336 \
  --model $model_name \
  --data ETTm2 \
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
  --e_layers 1 \
  --n_heads 16 \
  --d_model 32 \
  --d_ff 2048 \
  --batch_size 64 \
  --dropout 0.0214 \
  --period_len 24 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/ETT-small \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_720 \
  --model $model_name \
  --data ETTm2 \
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
  --n_heads 8 \
  --d_model 32 \
  --d_ff 256 \
  --batch_size 64 \
  --dropout 0.3262 \
  --period_len 12 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/ETT-small \
  --data_path ETTm2.csv \
  --model_id ETTm2_720_720 \
  --model $model_name \
  --data ETTm2 \
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
  --n_heads 8 \
  --d_model 32 \
  --d_ff 2048 \
  --batch_size 64 \
  --dropout 0.3043 \
  --period_len 24 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/ETT-small \
  --data_path ETTm2.csv \
  --model_id ETTm2_720_1440 \
  --model $model_name \
  --data ETTm2 \
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
  --e_layers 2 \
  --n_heads 16 \
  --d_model 32 \
  --d_ff 512 \
  --batch_size 64 \
  --dropout 0.4120 \
  --period_len 24 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/ETT-small \
  --data_path ETTm2.csv \
  --model_id ETTm2_1440_1440 \
  --model $model_name \
  --data ETTm2 \
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
  --e_layers 1 \
  --n_heads 4 \
  --d_model 32 \
  --d_ff 256 \
  --batch_size 64 \
  --dropout 0.3768 \
  --period_len 12 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1