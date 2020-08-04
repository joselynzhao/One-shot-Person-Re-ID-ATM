#dataset=market1501
#dataset=duke
#dataset=mars
dataset=DukeMTMC-VideoReID

EF=10
init=-1
loss=ExLoss
#loss=CrossEntropyLoss

fea=2048
momentum=0.5
epochs=70
stepSize=55
batchSize=64
lambda=0.8
max_frames=100
exp_name=atm
exp_order=1
#logs=logs/$dataset

python3.6  run.py --dataset $dataset  --max_frames $max_frames --EF $EF --init $init --loss $loss --fea $fea -m $momentum -e $epochs -s $stepSize -b $batchSize --lamda $lambda --exp_order $exp_order --exp_name $exp_name


# 多GPU distributedParallel 方法
#python3.6  -m torch.distributed.launch --nproc_per_node=4 run.py
# 测试另一个方法.
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3.6  -m torch.distributed.launch --nproc_per_node=4  run.py