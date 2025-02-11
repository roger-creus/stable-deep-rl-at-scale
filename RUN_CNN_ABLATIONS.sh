# # Normal trunk. Ablation of CNNs
# python src/pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=atari --total_timesteps=10000000
# python src/pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=impala --total_timesteps=10000000
# python src/pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=dense_residual --total_timesteps=10000000

# # Residual trunk. Ablation of CNNs
# python src/pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=atari --mlp_type=residual --total_timesteps=10000000
# python src/pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=impala --mlp_type=residual --total_timesteps=10000000
# python src/pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=dense_residual --mlp_type=residual --total_timesteps=10000000

# # MultiSkip trunk. Ablation of CNNs
# python src/pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=atari --mlp_type=multiskip_residual --total_timesteps=10000000
# python src/pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=impala --mlp_type=multiskip_residual --total_timesteps=10000000
# python src/pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=dense_residual --mlp_type=multiskip_residual --total_timesteps=10000000

# #### ABLATION AT SCALE DEPTH ####

# # Normal trunk. Ablation of CNNs
# python src/pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=atari --mlp_depth=xlarge --total_timesteps=10000000
# python src/pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=impala --mlp_depth=xlarge --total_timesteps=10000000
# python src/pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=dense_residual --mlp_depth=xlarge --total_timesteps=10000000

# # Residual trunk. Ablation of CNNs
# python src/pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=atari --mlp_type=residual --mlp_depth=xlarge --total_timesteps=10000000
# python src/pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=impala --mlp_type=residual --mlp_depth=xlarge --total_timesteps=10000000
# python src/pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=dense_residual --mlp_type=residual --mlp_depth=xlarge --total_timesteps=10000000

# # MultiSkip trunk. Ablation of CNNs
# python src/pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=atari --mlp_type=multiskip_residual --mlp_depth=xlarge --total_timesteps=10000000
# python src/pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=impala --mlp_type=multiskip_residual --mlp_depth=xlarge --total_timesteps=10000000
# python src/pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=dense_residual --mlp_type=multiskip_residual --mlp_depth=xlarge --total_timesteps=10000000


################################################ HL GAUSS PQN ################################################

# Normal trunk. Ablation of CNNs
python src/hlgauss_pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=impala --total_timesteps=10000000
python src/hlgauss_pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=atari --total_timesteps=10000000
python src/hlgauss_pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=dense_residual --total_timesteps=10000000

# Residual trunk. Ablation of CNNs
python src/hlgauss_pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=atari --mlp_type=residual --total_timesteps=10000000
python src/hlgauss_pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=impala --mlp_type=residual --total_timesteps=10000000
python src/hlgauss_pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=dense_residual --mlp_type=residual --total_timesteps=10000000

# MultiSkip trunk. Ablation of CNNs
python src/hlgauss_pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=atari --mlp_type=multiskip_residual --total_timesteps=10000000
python src/hlgauss_pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=impala --mlp_type=multiskip_residual --total_timesteps=10000000
python src/hlgauss_pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=dense_residual --mlp_type=multiskip_residual --total_timesteps=10000000

#### ABLATION AT SCALE DEPTH ####

# Normal trunk. Ablation of CNNs
python src/hlgauss_pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=atari --mlp_depth=xlarge --total_timesteps=10000000
python src/hlgauss_pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=impala --mlp_depth=xlarge --total_timesteps=10000000
python src/hlgauss_pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=dense_residual --mlp_depth=xlarge --total_timesteps=10000000

# Residual trunk. Ablation of CNNs
python src/hlgauss_pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=atari --mlp_type=residual --mlp_depth=xlarge --total_timesteps=10000000
python src/hlgauss_pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=impala --mlp_type=residual --mlp_depth=xlarge --total_timesteps=10000000
python src/hlgauss_pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=dense_residual --mlp_type=residual --mlp_depth=xlarge --total_timesteps=10000000

# MultiSkip trunk. Ablation of CNNs
python src/hlgauss_pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=atari --mlp_type=multiskip_residual --mlp_depth=xlarge --total_timesteps=10000000
python src/hlgauss_pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=impala --mlp_type=multiskip_residual --mlp_depth=xlarge --total_timesteps=10000000
python src/hlgauss_pqn.py --env_id=SpaceInvaders-v5 --wandb_project_id=multiskip_residual --cnn_type=dense_residual --mlp_type=multiskip_residual --mlp_depth=xlarge --total_timesteps=10000000