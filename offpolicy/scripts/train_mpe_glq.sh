#!/bin/sh
env="MPE"
scenario="simple_spread"
num_landmarks=3
num_agents=3
algo="glq"
exp="V1.4_0"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python3 train/train_mpe.py --num_landmarks 3 --num_agents 3 --env_name "MPE" --algorithm_name "glq" --experiment_name "V2.300_glqmix" --scenario_name "simple_spread" --seed 1 --episode_length 25 --batch_size 32 --tau 0.005 --lr 7e-4 --hard_update_interval_episode 100 --num_env_steps 7000000 --use_reward_normalization
    echo "training is done!"
done
