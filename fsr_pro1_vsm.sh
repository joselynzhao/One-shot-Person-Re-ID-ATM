batch_size=16
dataset=market1501
epochs=0
exp_aim=测试
exp_name=fsr
exp_order=test_vsm
fea=2048
init=-1.0
lamda=0.8
log_name=pl_logs
loss=ExLoss
max_frames=100
momentum=0.5
resume=Yes
run_file=fsr_pro1_vsm.py
step_size=55
t=1.5
EF=10
topk=2
vsm_lambda=0.5
#kf新增.
## yml:新增的参数
experiment='base'
##
python3.6  $run_file --dataset $dataset  --max_frames $max_frames --EF $EF --init $init --loss $loss --fea $fea -m $momentum -e $epochs -s $step_size -b $batch_size --lamda $lamda --experiment $experiment --exp_order $exp_order --exp_name $exp_name --exp_aim $exp_aim  --log_name $log_name --run_file $run_file --t $t --topk $topk --vsm_lambda $vsm_lambda
