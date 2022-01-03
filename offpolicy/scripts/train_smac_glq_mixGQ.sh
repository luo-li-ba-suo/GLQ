#!/bin/sh
env="StarCraft2"
maps="3m"
algo="glq_mixGQ"
exp="sn_NegativeHitPoint"
seeds="1"

echo "env is ${env}, map is ${maps}, algo is ${algo}, exp is ${exp}, seed_list is ${seeds}"

for map in $maps; do
    for seed in $seeds; do
        echo "map is ${map}:"
        echo "seed is ${seed}:"
        pip3 install -e ../..
        CUDA_VISIBLE_DEVICES=2 python3 train/train_smac.py --env_name ${env} \
         --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
          --seed 1 --n_training_threads 1 --buffer_size 5000 --lr 5e-4 --batch_size 32 --use_soft_update \
           --hard_update_interval_episode 200 --num_env_steps 8000000 \
           --log_interval 3000 --eval_interval 20000 --user_name "dujinqi"\
           --use_global_all_local_state --gain 1
        echo "training is done!"
    done
done

