###
 # @Author: y@yml.red
 # @Date: 1970-01-01 08:00:00
 # @LastEditTime: 2020-08-28 12:41:26
 # @FilePath: /One-shot-Person-Re-ID-ATM/run_atm_fsr.sh
 # @Description: 
### 
batch_size=16
dataset=market1501
epochs=0
exp_aim=测试pro12在fsr上的效果
exp_name=fsr
exp_order=fsr_atmkf3_pro12
fea=2048
init=-1.0
lamda=0.8
log_name=pl_logs
loss=ExLoss
max_frames=100
momentum=0.5
resume=Yes
run_file=fsr_atmkf3_pro12.py
step_size=55
t=1.5
EF=10
#kf新增.
gap=0.5
## yml:新增的参数
experiment='base'
##
python3.6  $run_file --dataset $dataset  --max_frames $max_frames --EF $EF --init $init --loss $loss --fea $fea -m $momentum -e $epochs -s $step_size -b $batch_size --lamda $lamda --experiment $experiment --exp_order $exp_order --exp_name $exp_name --exp_aim $exp_aim  --log_name $log_name --run_file $run_file --t $t --gap $gap
#python3.6  evaluate.py --dataset $dataset  --max_frames $max_frames --EF $EF --init $init --loss $loss --fea $fea -m $momentum -e $epochs -s $step_size -b $batch_size --lamda $lamda --exp_order $exp_order --exp_name $exp_name --exp_aim $exp_aim --log_name $log_name --run_file $run_file

# #evalueate
# dataset=mars
# save_path=/mnt/home/pl_logs/mars/atm/atm_0
# python3.6 evaluate_simple.py --dataset $dataset  --save_path $save_path


##dataset=market1501
##dataset=duke
##dataset=mars
#dataset=DukeMTMC-VideoReID
#
#EF=10
#init=-1
#loss=ExLoss
##loss=CrossEntropyLoss
#
#fea=2048
#momentum=0.5
#epochs=2
#stepSize=55
#batchSize=16
#lambda=0.8  #最后15轮会变成1  也就是不再考虑 最大化任意两个样本之间的距离
#max_frames=100
#exp_name=atm
#exp_order=test_toatal_step #baseline
#exp_aim=   ##可以在这里说明实验目的.
##logs=logs/$dataset