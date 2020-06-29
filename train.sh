OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/albu_example/faster_rcnn_mdconv_c3-c5_r50_fpn_1x.py 4 # 4为进程数
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/albu_example/faster_rcnn_dconv_c3-c5_x101_64x4d_fpn_1x.py 4
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 tools/train.py configs/albu_example/faster_rcnn_dconv_c3-c5_x101_64x4d_fpn_1x.py --gpus 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train.py configs/chest/mask_rcnn_r50_fpn_1x.py --gpus 4

# fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh

# 分布式训练: (1)中断时 会出现显存依旧存留的问题 进程也会残留 (2) 进程通信需要专门的通信端口

# 带有mask时
# 负样本 非分布式 单卡可训练 多卡无法训练
# 负样本 分布式 无法训练
# 第一次上传测试 
