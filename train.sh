CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset pascal_voc --net res101 --bs 1 --nw 1 --lr 0.001 --lr_decay_step 3000 --cuda
