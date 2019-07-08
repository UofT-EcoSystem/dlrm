#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#WARNING: must have compiled PyTorch and caffe2

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
echo "dlrm_extra_option = ${dlrm_extra_option}"

cpu=1
gpu=1

ncores=32
nsockets="0"

ngpus="1" #"1 2 4 8"

numa_cmd="numactl --physcpubind=0-$((ncores-1)) -m $nsockets" #run on one socket, without HT
dlrm_pt_bin="python dlrm_s_pytorch.py"

data=dataset
data_set=kaggle
print_freq=1024
rand_seed=727

#Model param
mb_size=128
bot_mlp="13-512-256-64-16"
top_mlp="512-256-1"
emb_size=16
loss=bce
round_targets=True
lr=0.1

_args="--data-generation="${data}\
" --data-set="${data_set}\
" --arch-mlp-bot="${bot_mlp}\
" --arch-mlp-top="${top_mlp}\
" --arch-sparse-feature-size="${emb_size}\
" --print-freq="${print_freq}\
" --loss-function="${loss}\
" --round-targets="${round_targets}\
" --learning-rate="${lr}\
" --print-time"\
" --enable-profiling "

# CPU Benchmarking
if [ $cpu = 1 ]; then
  echo "--------------------------------------------"
  echo "CPU Benchmarking - running on $ncores cores"
  echo "--------------------------------------------"

  outf="model1_CPU_PT_$ncores.log"
  outp="dlrm_s_pytorch.prof"
  echo "-------------------------------"
  echo "Running PT (log file: $outf)"
  echo "-------------------------------"
  cmd="$numa_cmd $dlrm_pt_bin --mini-batch-size=$mb_size $_args $dlrm_extra_option > $outf"
  echo $cmd
  eval $cmd
  min=$(grep "iteration" $outf | awk 'BEGIN{best=999999} {if (best > $7) best=$7} END{print best}')
  echo "Min time per iteration = $min"
  # move profiling file(s)
  mv $outp ${outf//".log"/".prof"}
  mv ${outp//".prof"/".json"} ${outf//".log"/".json"}
fi

# GPU Benchmarking
if [ $gpu = 1 ]; then
  echo "--------------------------------------------"
  echo "GPU Benchmarking - running on $ngpus GPUs"
  echo "--------------------------------------------"
  for _ng in $ngpus
  do
    # weak scaling
    # _mb_size=$((mb_size*_ng))
    # strong scaling
    _mb_size=$((mb_size*1))
    _gpus=$(seq -s, 0 $((_ng-1)))
    cuda_arg="CUDA_VISIBLE_DEVICES=$_gpus"
    echo "-------------------"
    echo "Using GPUS: "$_gpus
    echo "-------------------"

    outf="model1_GPU_PT_$_ng.log"
    outp="dlrm_s_pytorch.prof"
    echo "-------------------------------"
    echo "Running PT (log file: $outf)"
    echo "-------------------------------"
    cmd="$cuda_arg $dlrm_pt_bin --mini-batch-size=$_mb_size $_args --use-gpu $dlrm_extra_option > $outf"
    echo $cmd
    eval $cmd
    min=$(grep "iteration" $outf | awk 'BEGIN{best=999999} {if (best > $7) best=$7} END{print best}')
    echo "Min time per iteration = $min"
    # move profiling file(s)
    mv $outp ${outf//".log"/".prof"}
    mv ${outp//".prof"/".json"} ${outf//".log"/".json"}
  done
fi

echo "done"
