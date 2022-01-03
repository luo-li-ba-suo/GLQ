#!/bin/sh
env="MPE"
scenario="simple_spread"
num_landmarks=3
num_agents=3
algo="glq_Qbias"
exp="simplified_sn"
seed_max=5

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python3 train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --episode_length 25 --batch_size 32 --tau 0.005 --lr 7e-4 --hard_update_interval_episode 100 --num_env_steps 8000000 --use_same_share_obs --share_hyper_network --use_reward_normalization
    echo "training is done!"
done
