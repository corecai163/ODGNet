#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=13200 --nproc_per_node=2 main.py --launcher pytorch --sync_bn --config ./cfgs/ShapeNet55_models/UpTrans.yaml --resume --exp_name shape55_upTrans --num_workers 2 --val_freq 10 --val_interval 100 
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=13200 --nproc_per_node=2 main.py --launcher pytorch --sync_bn --config ./cfgs/ShapeNet55_models/UpTrans.yaml --exp_name shape55_upTransb1 --num_workers 2 --val_freq 10 --val_interval 100 