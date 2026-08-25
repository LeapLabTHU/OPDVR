#!/bin/bash
# ============================================================================
# OPDVR: On-policy Distillation with Verifiable Reward
#   sampled-token OPD + ReLU gating by trajectory correctness (the paper method)
#
#   reward(o_t) =  correct traj:  relu( log pi_T/pi_s )          (non-negative)
#                incorrect traj: -relu( log pi_s/pi_T )          (non-positive)
#
# Usage:  bash opdvr.sh          (override any env var, e.g. N_RESPONSES=8 ...)
# Requires: 8x GPUs, conda env `verl` (see README)
# ============================================================================

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# ---- cluster env ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export VERL_USE_MODELSCOPE=${VERL_USE_MODELSCOPE:-False}

ray stop --force; ray start --head; sleep 5

# ============================================================================
# Method switches (the OPDVR knobs added on top of verl)
# ============================================================================
export ADV_ESTIMATOR=token_reward_direct
export TEMPERATURE=${TEMPERATURE:-1.0}              # student sampling temperature
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-0}          # 0 = sampled-token OPD (paper); >0 = top-k OPD
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-"only_stu"}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-"student_p"}

# *** OPDVR: the ReLU gate ***
export CORRECTNESS_GATED=${CORRECTNESS_GATED:-False}
export CORRECTNESS_THRESHOLD=${CORRECTNESS_THRESHOLD:-0.0}
export CORRECTNESS_GATED_MODE=${CORRECTNESS_GATED_MODE:-default}  # default | inverse (ablation)

# *** GRPD: group-relative variant (Section "GRPD" in paper) ***
export GRPO_SCALED=${GRPO_SCALED:-False}             # True = GRPD (needs N_RESPONSES>1)
export GRPO_NORM_BY_STD=${GRPO_NORM_BY_STD:-False}   # False = Dr.GRPO style (paper); True = /std
export GRPO_SCALE_BASELINE=${GRPO_SCALE_BASELINE:-0.0}  # u in scale=|A|+u

# ---- shared hypers ----
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-8192}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-8192}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-256}
export N_RESPONSES=${N_RESPONSES:-1}                 # group size for GRPD (paper uses 8)
export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
export PARALLEL_SIZE=${PARALLEL_SIZE:-1}
export USE_KL=${USE_KL:-False}

# ---- models / data (EDIT THESE) ----
export TRAIN_DATASET=${TRAIN_DATASET:-"$SCRIPT_DIR/datasets/deepmath-level6-train.parquet"}
export TEST_DATASET=${TEST_DATASET:-"$SCRIPT_DIR/datasets/valid_final_unique.parquet"}
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-/cluster/data4/linwenze/models/Qwen/Qwen3-4B}     # student
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-/cluster/data4/linwenze/models/Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}          # teacher (reward model)

export PROJECT_NAME=${PROJECT_NAME:-OPDVR}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_${LOG_PROB_TOP_K}_${ACTOR_MODEL_PATH##*/}_$(date +%Y%m%d_%H%M%S)}
export FINAL_CKPT_DIR=${FINAL_CKPT_DIR:-checkpoints/${EXPERIMENT_NAME}}
mkdir -p "${FINAL_CKPT_DIR}"

MIN_TOKEN_LEN_PER_GPU=$(( MAX_PROMPT_LENGTH + MAX_RESP_LENGTH ))
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}
PPO_MAX_TOKEN_LEN_PER_GPU=$(( PPO_MAX_TOKEN_LEN_PER_GPU > MIN_TOKEN_LEN_PER_GPU ? PPO_MAX_TOKEN_LEN_PER_GPU : MIN_TOKEN_LEN_PER_GPU ))
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$MIN_TOKEN_LEN_PER_GPU}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=$(( LOG_PROB_MAX_TOKEN_LEN_PER_GPU > MIN_TOKEN_LEN_PER_GPU ? LOG_PROB_MAX_TOKEN_LEN_PER_GPU : MIN_TOKEN_LEN_PER_GPU ))

KL_ARGS=""
if [ "$USE_KL" = "True" ]; then
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.005 actor_rollout_ref.actor.kl_loss_type=low_var_kl"
else
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.00 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size=$((${MINI_BATCH_SIZE}*${PARALLEL_SIZE})) \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6} \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    $KL_ARGS \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.dtype=$MODEL_DTYPE \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    +actor_rollout_ref.rollout.log_prob_top_k=$LOG_PROB_TOP_K \
    +actor_rollout_ref.rollout.top_k_strategy=$TOP_K_STRATEGY \
    +actor_rollout_ref.rollout.reward_weight_mode=$REWARD_WEIGHT_MODE \
    +actor_rollout_ref.rollout.teacher_temperature=$TEACHER_TEMPERATURE \
    +actor_rollout_ref.rollout.correctness_gated=$CORRECTNESS_GATED \
    +actor_rollout_ref.rollout.correctness_threshold=$CORRECTNESS_THRESHOLD \
    +actor_rollout_ref.rollout.correctness_gated_mode=$CORRECTNESS_GATED_MODE \
    +actor_rollout_ref.rollout.grpo_scaled=$GRPO_SCALED \
    +actor_rollout_ref.rollout.grpo_norm_by_std=$GRPO_NORM_BY_STD \
    +actor_rollout_ref.rollout.grpo_scale_baseline=$GRPO_SCALE_BASELINE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESP_LENGTH)) \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    reward_model.enable=True \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype=$MODEL_DTYPE \
    reward_model.micro_batch_size_per_gpu=8 \
    custom_reward_function.path="verl/verl/utils/reward_score/__init__.py" \
    custom_reward_function.name=default_compute_score \
    trainer.val_before_train=False \
    trainer.log_val_generations=2 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE:-8} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-55} \
    trainer.test_freq=${TEST_FREQ:-22} \
    trainer.total_epochs=${TOTAL_EPOCHS:-3} \
    trainer.default_local_dir="$FINAL_CKPT_DIR" "$@"
