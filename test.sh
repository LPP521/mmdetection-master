
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_test.sh \
configs/albu_example/faster_rcnn_r50_fpn_1x.py \
./work_dirs/faster_rcnn_r50_fpn_1x/10_noneg_rotate/epoch_56.pth 4 \
--eval bbox --out output/out.pkl --overwrite

#CUDA_VISIBLE_DEVICES=4 python3 tools/test.py \
#configs/albu_example/faster_rcnn_dconv_c3-c5_x101_64x4d_fpn_1x.py \
#./work_dirs/faster_rcnn_dconv_c3-c5_x101_64x4d_fpn_1x/epoch_98.pth \
#--eval bbox --out output/out.pkl --overwrite

#CUDA_VISIBLE_DEVICES=4 python3 tools/test.py \
#configs/albu_example/mask_rcnn_dconv_c3-c5_r50_fpn_1x.py \
#./work_dirs/product/mask_rcnn_dconv_c3-c5_r50_fpn_1x/epoch_26.pth \
#--eval bbox --out output/out.pkl # --overwrite
