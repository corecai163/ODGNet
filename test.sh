#CUDA_VISIBLE_DEVICES=1 python main.py --test --ckpts experiments/UpTrans/ShapeNet34_models/512ckpt-best1.pth --config ./cfgs/ShapeNet34_models/UpTrans.yaml --mode easy --test_interval 50 --exp_name test_ckpt
#CUDA_VISIBLE_DEVICES=0 python main.py --test --ckpts experiments/UpTrans/ShapeNet34_models/512ckpt-best1.pth --config ./cfgs/ShapeNetUnseen21_models/UpTrans.yaml --mode hard --test_interval 50 --exp_name test_ckpt
#CUDA_VISIBLE_DEVICES=0 python main.py --test --ckpts experiments/UpTrans/ShapeNet55_models/ckpt-best.pth --config ./cfgs/ShapeNet55_models/UpTrans.yaml --mode hard --test_interval 50 --exp_name test_ckpt
#CUDA_VISIBLE_DEVICES=1 python main.py --test --ckpts experiments/UpTrans/ShapeNet55_models/ckpt-best.pth --config ./cfgs/ShapeNet55_models/UpTrans.yaml --mode easy --test_interval 50 --exp_name test_ckpt
CUDA_VISIBLE_DEVICES=0 python main.py --test --ckpts experiments/UpTrans/PCN_models/ckpt-best.pth --config ./cfgs/PCN_models/UpTrans.yaml --test_interval 50 --exp_name test_ckpt
#CUDA_VISIBLE_DEVICES=1 python main.py --test --ckpts experiments/UpTrans/KITTI_models/ckpt-best.pth --config ./cfgs/KITTI_models/UpTrans.yaml --mode easy --test_interval 50 --exp_name test_ckpt
#CUDA_VISIBLE_DEVICES=1 python main.py --test --ckpts experiments/UpTrans/PCN_models/ckpt-best.pth --config ./cfgs/PCN_models/UpTrans.yaml --mode easy --test_interval 50 --exp_name test_ckpt
