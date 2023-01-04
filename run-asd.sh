#!/usr/bin/env bash
#SBATCH --job-name=yichenggu_test1        # Job name
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=1                   # Run a single task		
#SBATCH --cpus-per-task=6            # Number of CPU cores per task
# #SBATCH --output=./summit_log/summit_%j.log       # Standard output and error log
#SBATCH --gres=gpu:1                # Number of GPU cores per task
#SBATCH --nodelist=pgpu04
# #SBATCH --exclude=pgpu14,pgpu17,pgpu22
#SBATCH --partition=p-RTX2080

source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate guyicheng
cd /mntnfs/lee_data1/guyicheng/SVC-baseline-model/

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Number of CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo ""
echo "Running script... "

bash run.sh train.py
# 13 18 = available

#####------------------------cnn module---------------------------#####
# model=dla_raw_small # dla_raw resnet_raw resnet_lfcc resnet_mfcc vgg resnet googlenet dla efficientnet mobilenet resnet_chunk16 resnet_chunk8 resnet_chunk4 resnet_chunk64
# model=dla_raw_small_av_ao
# model=dla_raw_small_av
# echo "Model: ${model}"

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 ASD/bin/train.py \
# python ASD/bin/train.py \
# --config ./ASD/conf/train_${model}.yaml --train_data ./ASD/data/train/data.list  --cv_data ./ASD/data/dev/data.list \
# --gpu 0,1 --model_dir /mntnfs/lee_data1/lizuoou/AVSD_cache/${model} --num_workers 6 --pin_memory

# python visualize.py
# bash inference.sh

# data="test"
# echo "----- Testing ${model} in ${data} set -----"
# python ASD/bin/inference.py --config ./ASD/conf/train_${model}.yaml --gpu 0 \
#         --test_data ./ASD/data/${data}/data.list --checkpoint /mntnfs/lee_data1/lizuoou/AVSD_cache/${model}/best_f1.pt > inf_${model}_sft${shift}_${data}.txt
