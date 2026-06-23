nohup accelerate launch --multi_gpu --num_processes 2 --main_process_port 29502 /home/Yb/uureal/trainVAL.py  &
 CUDA_VISIBLE_DEVICES=1 nohup --multi_gpu --num_processes 1 --main_process_port 29502 /home/Yb/uureal/trainVAL.py  &
 nohup accelerate launch --multi_gpu --num_processes 2 --main_process_port 29502 /home/Yb/uureal/tg.py  &