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
batchSize=16
lambda=0.8
max_frames=100
exp_name=atm
exp_order=0
#logs=logs/$dataset

#python3.6 -m torch.distributed.launch --nproc_per_node=4 run.py --dataset $dataset  --max_frames $max_frames --logs_dir $logs --EF $EF --init $init --loss $loss --fea $fea -m $momentum -e $epochs -s $stepSize -b $batchSize --lamda $lambda
python3.6  run.py --dataset $dataset  --max_frames $max_frames --EF $EF --init $init --loss $loss --fea $fea -m $momentum -e $epochs -s $stepSize -b $batchSize --lamda $lambda --exp_order $exp_order --exp_name $exp_name
