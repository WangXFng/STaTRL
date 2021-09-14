device=0
# data=data/data_so/fold1/
data=./data/Yelp/
batch=8
n_head=4
n_layers=4
d_model=512  # M
d_rnn=128
d_inner=1024 # H 1024
d_k=512   # Mk
d_v=512   # Mv
dropout=0.1
lr=1e-4
smooth=0.1
epoch=30
log=log.txt

CUDA_VISIBLE_DEVICES=$device python Main.py -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -log $log
