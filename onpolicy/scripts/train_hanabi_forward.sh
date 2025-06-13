#!/bin/sh
env="Hanabi"
hanabi="Hanabi-Full"
num_agents=2
algo="rmappo"
exp="hanabi"
seed_start=180
num_seeds=5
ulimit -n 22222

# pruning methods and schedules
pruning_methods="gradual_schedule_l1 none"
schedule_types="linear cyclical polynomial"

echo "env is ${env}, hanabi game is ${hanabi}, algo is ${algo}, exp is ${exp}, num seeds is ${num_seeds}"
for pruning_method in $pruning_methods; do
    if [ "$pruning_method" = "none" ]; then
        for seed in $(seq $seed_start $(($seed_start + $num_seeds - 1))); do
            echo "pruning method is ${pruning_method}, seed is ${seed}:"
            CUDA_VISIBLE_DEVICES=0 python train/train_hanabi_forward.py \
            --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
            --hanabi_name ${hanabi} --num_agents ${num_agents} --seed ${seed} \
            --n_training_threads 1 --n_rollout_threads 1000 \
            --num_mini_batch 1 --episode_length 100 --num_env_steps 100000000 \
            --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 1e-3 \
            --wandb_name "akanatas" --user_name "MARL-pruning" \
            --hidden_size 512 --layer_N 2 --entropy_coef 0.015 \
            --pruning_method "none"
        done
    else
        for schedule_type in $schedule_types; do
            for seed in $(seq $seed_start $(($seed_start + $num_seeds - 1))); do
                echo "pruning method is ${pruning_method}, schedule is ${schedule_type}, seed is ${seed}:"
                CUDA_VISIBLE_DEVICES=0 python train/train_hanabi_forward.py \
                --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
                --hanabi_name ${hanabi} --num_agents ${num_agents} --seed ${seed} \
                --n_training_threads 1 --n_rollout_threads 1000 \
                --num_mini_batch 1 --episode_length 100 --num_env_steps 100000000 \
                --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 1e-3 \
                --wandb_name "akanatas" --user_name "MARL-pruning" \
                --hidden_size 512 --layer_N 2 --entropy_coef 0.015 \
                --pruning_method ${pruning_method} --schedule_type ${schedule_type} \
                --initial_sparsity 0.0 --final_sparsity 0.95 --warmup_episodes 0 --prune_interval 5 --endlock_episodes 100
            done
        done
    fi
done

seed=$seed_start
echo "pruning method is gradual_schedule_random, schedule is linear, seed is ${seed}:"
CUDA_VISIBLE_DEVICES=0 python train/train_hanabi_forward.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --hanabi_name ${hanabi} --num_agents ${num_agents} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1000 \
    --num_mini_batch 1 --episode_length 100 --num_env_steps 100000000 \
    --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 1e-3 \
    --wandb_name "akanatas" --user_name "MARL-pruning" \
    --hidden_size 512 --layer_N 2 --entropy_coef 0.015 \
    --pruning_method "gradual_schedule_random" --schedule_type "linear" \
    --initial_sparsity 0.0 --final_sparsity 0.95 --warmup_episodes 0 --prune_interval 5 --endlock_episodes 100

# ------------------------------------------
# HARMONIC PRUNING ONLY
# ------------------------------------------

echo "Running harmonic pruning experiments..."
exp="hanabi_harmonic"

pruning_method="gradual_schedule_l1"
schedule_type="harmonic"
harmonic_base_schedule_type="linear"
harmonic_A0=0.1
harmonic_lambda_decay=0.0
harmonic_T0=100
harmonic_T_increase_rate=0.0

echo "env is ${env}, hanabi game is ${hanabi}, algo is ${algo}, exp is ${exp}, num seeds is ${num_seeds}"
for seed in $(seq $seed_start $(($seed_start + $num_seeds - 1))); do
    echo "pruning method is ${pruning_method}, schedule is harmonic, seed is ${seed}:"
    echo "Running Harmonic Pruning:"
    echo "  base_schedule: ${harmonic_base_schedule_type}"
    echo "  A0: ${harmonic_A0}, lambda_decay: ${harmonic_lambda_decay}, T0: ${harmonic_T0}, T_rate: ${harmonic_T_increase_rate}"
    CUDA_VISIBLE_DEVICES=0 python train/train_hanabi_forward.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --hanabi_name ${hanabi} --num_agents ${num_agents} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1000 \
    --num_mini_batch 1 --episode_length 100 --num_env_steps 100000000 \
    --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 1e-3 \
    --wandb_name "akanatas" --user_name "MARL-pruning" \
    --hidden_size 512 --layer_N 2 --entropy_coef 0.015 \
    --pruning_method ${pruning_method} --schedule_type ${schedule_type} \
    --initial_sparsity 0.0 --final_sparsity 0.95 --warmup_episodes 0 --prune_interval 5 --endlock_episodes 100 \
    --harmonic_base_schedule_type ${harmonic_base_schedule_type} \
    --harmonic_A0 ${harmonic_A0} \
    --harmonic_lambda_decay ${harmonic_lambda_decay} \
    --harmonic_T0 ${harmonic_T0} \
    --harmonic_T_increase_rate ${harmonic_T_increase_rate}
done
