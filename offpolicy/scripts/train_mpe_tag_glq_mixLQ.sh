#!/bin/sh
env="MPE"
scenario="joint_tag"
num_landmarks=2
num_good_agents=1
num_adversaries=3
algo="glq_mixLQ"
exp="nso_sh"
seed_max=3

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_max}); do
    echo "seed is ${seed}:"
    echo "num_landmarks_ is ${num_landmarks}:"
    echo "num_good_agents is ${num_good_agents}:"
    echo "num_adversaries is ${num_adversaries}:"
    CUDA_VISIBLE_DEVICES=0 python3 train/train_mpe.py --env_name ${env} --algorithm_name ${algo} \
      --experiment_name ${exp} --scenario_name ${scenario} --num_good_agents ${num_good_agents} --num_landmarks \
      ${num_landmarks} --seed ${seed} --episode_length 25 --batch_size 32 --tau 0.005 --lr 7e-4 \
      --hard_update_interval_episode 100 --num_env_steps 5000000 --use_reward_normalization\
      --n_rollout_threads 8 --n_eval_rollout_threads 8 --save_all --num_adversaries ${num_adversaries}\
      --share_hyper_network
#      --ablation_resort_q\
#      --ablation_share_reward\
#      --use_same_share_obs
    echo "training is done!"
done