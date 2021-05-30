export LD_LIBRARY_PATH=/media/zhizhong.zhang/project/zzw:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py
ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
