batch_size=16
dataset=market1501
epochs=0
exp_aim=fsr原型
exp_name=fsr
exp_order=atm_0
fea=2048
init=-1.0
lamda=0.8
log_name=fsr_logs
loss=ExLoss
max_frames=100
momentum=0.5
resume=Yes
run_file=run_fsr.py
step_size=55
EF=10
## yml:新增的参数
experiment='base'
python3.6  $run_file --dataset $dataset  --max_frames $max_frames --EF $EF --init $init --loss $loss --fea $fea -m $momentum -e $epochs -s $step_size -b $batch_size --lamda $lamda --experiment $experiment --exp_order $exp_order --exp_name $exp_name --exp_aim $exp_aim  --log_name $log_name --run_file $run_file
