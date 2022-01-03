#!/bin/sh
env="StarCraft2"
maps="3m"
algo="rmaddpg"
exp="base"
seeds="1 2 3 4 5"

echo "env is ${env}, map_list is ${maps}, algo is ${algo}, exp is ${exp}, seed_list is ${seeds}"
for map in $maps; do
    for seed in $seeds; do
        echo "map is ${map}:"
        echo "seed is ${seed}:"
        pip3 install -e ../..
        CUDA_VISIBLE_DEVICES=2 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} --lr 5e-4 --buffer_size 5000 --batch_size 32 --actor_train_interval_step 1 --tau 0.005 --num_env_steps 5000000 --log_interval 20000 --user_name "dujinqi"
        echo "training is done!"
    done
done
