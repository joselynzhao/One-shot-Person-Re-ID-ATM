batch_size=16
dataset=market1501
epochs=70
exp_aim=atmpro1和加上vsm的效果做duibi
exp_name=vsm
exp_order=
fea=2048
init=-1.0
kf_thred=0.5
lamda=0.8
log_name=pl_logs
loss=ExLoss
max_frames=100
momentum=0.5
resume=Yes
run_file=atmpro1_vsm2.py
step_size=55
EF=10
t=2
topk=2
vsm_lambda=0.5

####
python3.6  $run_file --dataset $dataset  --max_frames $max_frames --EF $EF --init $init --loss $loss --fea $fea -m $momentum -e $epochs -s $step_size -b $batch_size --lamda $lamda --exp_order $exp_order --exp_name $exp_name --exp_aim $exp_aim  --log_name $log_name --run_file $run_file --topk $topk --vsm_lambda $vsm_lambda --t $t
