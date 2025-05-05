python src/ppo_craftax.py  --track --mlp_depth=small --total_timesteps=250000000
python src/pqn_craftax.py  --track --mlp_depth=small --total_timesteps=250000000
#python src/ppo_lstm_craftax.py  --track --mlp_depth=small --total_timesteps=250000000
#python src/pqn_lstm_craftax.py  --track --mlp_depth=small --total_timesteps=250000000

python src/ppo_craftax.py  --optimizer=kron --mlp_type=multiskip_residual --use_ln --track --mlp_depth=small --total_timesteps=250000000
python src/pqn_craftax.py  --optimizer=kron --mlp_type=multiskip_residual --use_ln --track --mlp_depth=small --total_timesteps=250000000
#python src/ppo_lstm_craftax.py  --optimizer=kron --mlp_type=multiskip_residual --use_ln --track --mlp_depth=small --total_timesteps=250000000
#python src/pqn_lstm_craftax.py  --optimizer=kron --mlp_type=multiskip_residual --use_ln --track --mlp_depth=small --total_timesteps=250000000

python src/ppo_craftax.py  --track --mlp_depth=medium --total_timesteps=250000000
python src/pqn_craftax.py  --track --mlp_depth=medium --total_timesteps=250000000
#python src/ppo_lstm_craftax.py  --track --mlp_depth=medium --total_timesteps=250000000
#python src/pqn_lstm_craftax.py  --track --mlp_depth=medium --total_timesteps=250000000

python src/ppo_craftax.py  --optimizer=kron --mlp_type=multiskip_residual --use_ln --track --mlp_depth=medium --total_timesteps=250000000
python src/pqn_craftax.py  --optimizer=kron --mlp_type=multiskip_residual --use_ln --track --mlp_depth=medium --total_timesteps=250000000
#python src/ppo_lstm_craftax.py  --optimizer=kron --mlp_type=multiskip_residual --use_ln --track --mlp_depth=medium --total_timesteps=250000000
#python src/pqn_lstm_craftax.py  --optimizer=kron --mlp_type=multiskip_residual --use_ln --track --mlp_depth=medium --total_timesteps=250000000

python src/ppo_craftax.py  --track --mlp_depth=large --total_timesteps=250000000
python src/pqn_craftax.py  --track --mlp_depth=large --total_timesteps=250000000
#python src/ppo_lstm_craftax.py  --track --mlp_depth=large --total_timesteps=250000000
#python src/pqn_lstm_craftax.py  --track --mlp_depth=large --total_timesteps=250000000

python src/ppo_craftax.py  --optimizer=kron --mlp_type=multiskip_residual --use_ln --track --mlp_depth=large --total_timesteps=250000000
python src/pqn_craftax.py  --optimizer=kron --mlp_type=multiskip_residual --use_ln --track --mlp_depth=large --total_timesteps=250000000
#python src/ppo_lstm_craftax.py  --optimizer=kron --mlp_type=multiskip_residual --use_ln --track --mlp_depth=large --total_timesteps=250000000
#python src/pqn_lstm_craftax.py  --optimizer=kron --mlp_type=multiskip_residual --use_ln --track --mlp_depth=large --total_timesteps=250000000
