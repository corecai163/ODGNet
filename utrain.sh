CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python -m torch.distributed.launch --master_port=13231 --nproc_per_node=1 main.py --launcher pytorch --sync_bn --config ./cfgs/ShapeNet55_models/UpTrans.yaml --exp_name pcn_UpTrans_256 --val_freq 10 --val_interval 50
