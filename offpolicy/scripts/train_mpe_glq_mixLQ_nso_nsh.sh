#!/bin/sh
env="MPE"
scenario="simple_spread"
# shellcheck disable=SC2039
num_landmarks=(3 5 8)
# shellcheck disable=SC2039
num_agents=(3 5 8)
algo="glq_mixLQ"
exp="nso_nsh"
seed_max=3

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
# shellcheck disable=SC2128
# shellcheck disable=SC2068
# shellcheck disable=SC2039
for num_landmarks_ in ${num_landmarks[@]}; do
  # shellcheck disable=SC2034
  # shellcheck disable=SC2068
  # shellcheck disable=SC2039
  for num_agents_ in ${num_agents[@]}; do
    for seed in $(seq ${seed_max}); do
        echo "seed is ${seed}:"
        echo "num_landmarks_ is ${num_landmarks_}:"
        echo "num_agents_ is ${num_agents_}:"
        CUDA_VISIBLE_DEVICES=0 python3 train/train_mpe.py --env_name ${env} \--algorithm_name ${algo} \
          --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents_} --num_landmarks \
          ${num_landmarks_} --seed ${seed} --episode_length 25 --batch_size 32 --tau 0.005 --lr 7e-4 \
          --hard_update_interval_episode 100 --num_env_steps 5000000 --use_reward_normalization\
          --n_rollout_threads 8 --n_eval_rollout_threads 8
        echo "training is done!"
    done
  done
done
