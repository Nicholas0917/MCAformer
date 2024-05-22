if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
  mkdir ./logs/LongForecasting
fi
gpu=3
station_type=adaptive
model_name=MCAformer

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_168_168 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --label_len 0 \
  --pred_len 168 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 1 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 2048 \
  --batch_size 32 \
  --dropout 0.0281 \
  --period_len 12 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_168_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 168 \
  --label_len 0 \
  --pred_len 336 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 1 \
  --n_heads 4 \
  --d_model 32 \
  --d_ff 1024 \
  --batch_size 32 \
  --dropout 0.0361 \
  --period_len 24 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_336_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --label_len 0 \
  --pred_len 336 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 2 \
  --n_heads 4 \
  --d_model 32 \
  --d_ff 2048 \
  --batch_size 32 \
  --dropout 0.2781 \
  --period_len 24 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_336_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --label_len 0 \
  --pred_len 720 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 1 \
  --n_heads 4 \
  --d_model 256 \
  --d_ff 2048 \
  --batch_size 32 \
  --dropout 0.0844 \
  --period_len 12 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_720_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --label_len 0 \
  --pred_len 720 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 1 \
  --n_heads 16 \
  --d_model 32 \
  --d_ff 2048 \
  --batch_size 32 \
  --dropout 0.2928 \
  --period_len 24 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_720_1440 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --label_len 0 \
  --pred_len 1440 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 2 \
  --n_heads 16 \
  --d_model 32 \
  --d_ff 2048 \
  --batch_size 32 \
  --dropout 0.3805 \
  --period_len 12 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./datasets/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_1440_1440 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 1440 \
  --label_len 0 \
  --pred_len 1440 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --gpu $gpu \
  --station_type $station_type \
  --e_layers 2 \
  --n_heads 8 \
  --d_model 256 \
  --d_ff 2048 \
  --batch_size 32 \
  --dropout 0.1445 \
  --period_len 12 \
  --learning_rate 0.0001 \
  --station_lr 0.0001 \
  --itr 1