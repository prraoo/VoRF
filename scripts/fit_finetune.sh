#!/bin/bash
#SBATCH -p gpu16
#SBATCH -t 0-00:29:30
#SBATCH -o /HPS/prao2/work/NeRF/OLAT/batch-fit/logs/out-%j.out
#SBATCH -e /HPS/prao2/work/NeRF/OLAT/batch-fit/logs/err-%j.err
##SBATCH -a 1-24%24
#SBATCH --gres gpu:1

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT
# Make conda available:
eval "$(conda shell.bash hook)"

# go to base directory and activate environment
# conda activate nerf
cd /HPS/prao2/work/NeRF/OLAT/VoRF/code/
echo "using GPU ${CUDA_VISIBLE_DEVICES}"
echo "Working Directory: $PWD"
echo 'Start training'


# ID=(500)
# DATA_ID=(0)

ID=($1)
DATA_ID=($2)

#ID=(510 509 507 500)
#DATA_ID=(3 2 1 0)
# 

# DATA_DIR=/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/nerf-align-test-h3ds-1view/
# EXP_NAME=OLAT_c2_batch_OLATs_latent-300IDs-300EMAPs-relit-multi-gpu-shading-h3ds-1view/

# DATA_DIR=/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/nerf-align-test-h3ds-2views/
# EXP_NAME=OLAT_c2_batch_OLATs_latent-300IDs-300EMAPs-relit-multi-gpu-shading-h3ds-2views/

# DATA_DIR=/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/nerf-align-test-h3ds-3views/
# EXP_NAME=OLAT_c2_batch_OLATs_latent-300IDs-300EMAPs-relit-multi-gpu-shading-h3ds/

# DATA_DIR=/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/nerf-align-test-celeba-1view/
# EXP_NAME=OLAT_c2_batch_OLATs_latent-300IDs-300EMAPs-relit-multi-gpu-shading-celeba-1view/

# DATA_DIR=/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/nerf-align-test-celeba-1view/
# EXP_NAME=OLAT_c2_batch_OLATs_latent-300IDs-300EMAPs-relit-multi-gpu-shading-ffhq-1view/

# DATA_DIR=/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/nerf-align-test-i3DMM_all_expr/
# EXP_NAME=OLAT_c2_batch_OLATs_latent-300IDs-300EMAPs-relit-multi-gpu-shading-i3DMM_all_expr/


# DATA_DIR=/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/nerf-align-test-ravdess-1view/
# EXP_NAME=OLAT_c2_batch_OLATs_latent-300IDs-300EMAPs-relit-multi-gpu-shading-ravdess-1view/

# DATA_DIR=/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/nerf-align-test-3dpr-1view/
# EXP_NAME=OLAT_c2_batch_OLATs_latent-300IDs-300EMAPs-relit-multi-gpu-shading-3dpr-1view/

EXP_NAME=OLAT_c2_batch_OLATs_latent-50IDs-300EMAPs-relit-multi-gpu-256_latent_dim/
DATA_DIR=/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/nerf-align-test-h3ds-1view/


CONFIG_PATH=/HPS/prao2/work/NeRF/OLAT/VoRF/code/configs/all_OLATs-batch-h3ds.txt
PYPATH1=/HPS/prao2/work/anaconda3/envs/eg3d/bin/python
PYPATH0=/HPS/prao/work/anaconda3/envs/nerf/bin/python

# IDX=$(printf "%02d" $[SLURM_ARRAY_TASK_ID-1])

# ID=(5${IDX})
# DATA_ID=($[SLURM_ARRAY_TASK_ID-1])

if [[ $3 -eq 0 ]]
then
for ((i=0;i<${#ID[@]};i++))
do
    echo ${ID[$i]} ${DATA_ID[$i]};
    # srun -p gpu$4 -t 29:00 --gres gpu:1 \
    $PYPATH1 -m pdb -c cont run_nerf_batch_emap.py \
    --config $CONFIG_PATH \
    --i_video 3000000 --i_testset 1500 \
    --i_weights 1500 --i_print 1000 \
    --expname $EXP_NAME${ID[$i]} \
    --datadir $DATA_DIR${DATA_ID[$i]} \
    --lrate 0 \
    --fit \

done
# 
# for ((i=0;i<${#ID[@]};i++))
# do
#     echo ${ID[$i]} ${DATA_ID[$i]};
#     srun -p gpu20 -t 29:00 --gres gpu:1 \
#     /HPS/prao/work/anaconda3/envs/nerf/bin/python run_nerf_batch_emap.py \
#     --config $CONFIG_PATH \
#     --i_video 3000000 --i_testset 1000 \
#     --i_weights 500 --i_print 1000 \
#     --expname $EXP_NAME${ID[$i]} \
#     --datadir $DATA_DIR${DATA_ID[$i]} \
#     --lrate 0 \
# 
# done

# ID=(510 509 507 500)
# DATA_ID=(3 2 1 0)
# 
# DATA_DIR=/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/nerf-align-test-h3ds-3views/
# EXP_NAME=OLAT_c2_batch_OLATs_latent-300IDs-300EMAPs-relit-multi-gpu-shading/
# 
elif [[ $3 -eq 1 ]]
then
for ((i=0;i<${#ID[@]};i++))
do
    echo ${ID[$i]} ${DATA_ID[$i]};
    srun -p gpu$4 -t 19:00 --gres gpu:1 \
    $PYPATH1 run_nerf_batch_emap.py \
    --config $CONFIG_PATH \
    --i_video 3000000 --i_testset 1000 \
    --i_weights 1000 --i_print 1000 \
    --expname $EXP_NAME${ID[$i]} \
    --datadir $DATA_DIR${DATA_ID[$i]} \
    --lrate 5e-5 \

done
# 
# for ((i=0;i<${#ID[@]};i++))
# do
#     echo ${ID[$i]} ${DATA_ID[$i]};
#     srun -p gpu20 -t 59:00 --gres gpu:1 \
#     /HPS/prao/work/anaconda3/envs/nerf/bin/python run_nerf_batch_emap.py \
#     --config configs/all_OLATs-batch-h3ds.txt \
#     --i_video 3000000 --i_testset 1300 \
#     --i_weights 9000 --i_print 1000 \
#     --expname $EXP_NAME${ID[$i]} \
#     --datadir $DATA_DIR${DATA_ID[$i]} \
#     --lrate 5e-5 \
# 
# done
else
  echo "INVALID MODE"
fi

wait
