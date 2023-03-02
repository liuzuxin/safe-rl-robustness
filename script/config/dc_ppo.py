from ray import tune

ENV_LIST = [
    'SafetyDroneCircle-v0',
]

all_rs_mode = ["kl", "klmc", "klmr", "vanilla", "uniform", "max_cost", "max_reward"]

EXP_CONFIG = dict(
    env_name=tune.grid_search(ENV_LIST),
    cost_limit=tune.grid_search([5]),
    timeout_steps=300,
    policy_name=tune.grid_search(["robust_ppo_lag"]),
    rs_mode=tune.grid_search(all_rs_mode),
    kl_coef=1,
    noise_scale=0.025,
    train_actor_iters=80,
    seed=tune.grid_search([0, 11, 22, 33, 44]),
    clip_ratio=0.2,
    target_kl=0.01,
    epochs=500,
    save_best_epoch=400,
    interact_steps=60000,
    evaluate_episode_num=1,
    warmup_steps=0,
    batch_size=300,
    verbose=False,
    device="cpu",
    threads=4,
    hidden_sizes=[256, 256],
    gamma=0.99,
    decay_epoch=100,
    start_epoch=100,
)

SKIP_EXP_CONFIG = []

EXP_NAME_KEYS = {"rs_mode": "mode"}
DATA_DIR_KEYS = {"cost_limit": "cost"}
DATA_DIR_SUFFIX = '_benchmark_ppo'