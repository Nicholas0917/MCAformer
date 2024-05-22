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
  --root_path ./datasets/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_168_168 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --label_len 0 \
  --pred_len 168 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 2 \
  --n_heads 8 \
  --d_model 32 \
  --d_ff 4096 \
  --batch_size 32 \
  --dropout 0.2158 \
  --period_len 24 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_168_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --label_len 0 \
  --pred_len 336 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 3 \
  --n_heads 4 \
  --d_model 32 \
  --d_ff 4096 \
  --batch_size 64 \
  --dropout 0.3613 \
  --period_len 24 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_336_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --label_len 0 \
  --pred_len 336 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 1 \
  --n_heads 4 \
  --d_model 32 \
  --d_ff 4096 \
  --batch_size 64 \
  --dropout 0.3870 \
  --period_len 24 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_336_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --label_len 0 \
  --pred_len 720 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 1 \
  --n_heads 16 \
  --d_model 32 \
  --d_ff 4096 \
  --batch_size 16 \
  --dropout 0.3560 \
  --period_len 24 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_720_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --label_len 0 \
  --pred_len 720 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type 'none' \
  --e_layers 2 \
  --n_heads 16 \
  --d_model 32 \
  --d_ff 4096 \
  --batch_size 16 \
  --dropout 0.3080 \
  --period_len 24 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_720_1440 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --label_len 0 \
  --pred_len 1440 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 1 \
  --n_heads 32 \
  --d_model 512 \
  --d_ff 2048 \
  --batch_size 16 \
  --dropout 0.1 \
  --period_len 24 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_1440_1440 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 1440 \
  --label_len 0 \
  --pred_len 1440 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 2 \
  --n_heads 16 \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 16 \
  --dropout 0.1824 \
  --period_len 24 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1