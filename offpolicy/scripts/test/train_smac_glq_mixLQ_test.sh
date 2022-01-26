#!/bin/sh
env="StarCraft2"
maps="6h_vs_8z"
algo="glq_mixLQ"
exp="sn"
seeds="42"

echo "env is ${env}, map_list is ${maps}, algo is ${algo}, exp is ${exp}, seed_list is ${seeds}"
for map in $maps; do
    for seed in $seeds; do
        echo "map is ${map}:"
        echo "seed is ${seed}:"
        pip3 install -e ../../..
        CUDA_VISIBLE_DEVICES=0 python3 ../train/train_smac.py --env_name ${env} \
         --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
          --seed ${seed} --n_training_threads 1 --buffer_size 5000 --lr 5e-4 --batch_size 32 --use_soft_update \
           --hard_update_interval_episode 200 --num_env_steps 10000 \
           --log_interval 3000 --eval_interval 20000 --user_name "dujinqi"\
           --use_global_all_local_state --gain 1 --use_same_share_obs \
           --use_wandb --if_train --model_dir "smac/${algo}/${map}/"
        echo "training is done!"
    done
done

