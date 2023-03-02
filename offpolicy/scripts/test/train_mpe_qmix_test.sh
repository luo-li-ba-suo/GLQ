#!/bin/sh
env="MPE"
scenario="simple_spread"
num_landmarks=3
num_agents=3
algo="qmix"
exp="debug"
seed=100
model_idx=20000
model_num=250

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
echo "seed is ${seed}:"

# shellcheck disable=SC2034
for model_n in $(seq ${model_num}); do
  echo "Test model:""$model_idx"
  CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
    --num_landmarks ${num_landmarks} --seed ${seed} --episode_length 25 --batch_size 32 \
    --tau 0.005 --lr 7e-4 --hard_update_interval_episode 100 --num_env_steps 10000000 \
    --use_same_share_obs --use_reward_normalization\
    --use_wandb --if_train --model_name "$model_idx"".q_network.pt"\
    --model_dir "${env}/${scenario}/${algo}/" --render_interval 0.25
  echo "Testing is done!"
  # shellcheck disable=SC2039
  let model_idx=$model_idx+20000
done
