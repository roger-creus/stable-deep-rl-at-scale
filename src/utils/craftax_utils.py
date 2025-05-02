import os
import csv
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
from Craftax_Baselines.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)
import jax
import numpy as np
import gymnasium as gym
from dataclasses import asdict
from brax.io import torch as brax_torch
try:
    from craftax.craftax.constants import *
except:
    from craftax.constants import *
    
class TorchWrapper(gym.Wrapper):
    """Wrapper that converts Jax tensors to PyTorch tensors."""
    
    def __init__(self, env, device):
        self.device = device
        self.env = env
        self.default_params = env.default_params
        self.static_env_params = self.env.static_env_params
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
        }
        
        # define obs and action space
        obs_shape = env.observation_space(self.default_params).shape
        self.observation_space = gym.spaces.Box(
            low=-1e6, high=1e6, shape=obs_shape)
        self.action_space = gym.spaces.Discrete(env.action_space(self.default_params).n)

        # jit the reset function
        def reset(key):
            key1, key2 = jax.random.split(key)
            obs, state = self.env.reset(key2)
            return state, obs, key1, asdict(state)
        self._reset = jax.jit(reset)

        # jit the step function
        def step(key, state, action):
            obs, env_state, reward, done, info = self.env.step(rng=key, state=state, action=action)
            return env_state, obs, reward, done, {**asdict(env_state), **info}
        self._step = jax.jit(step)

    def reset(self, seed=0, options=None):
        random_seed = np.random.randint(0, 2**32)
        self.seed(random_seed)
        self._state, obs, self._key, info = self._reset(self._key)
        return brax_torch.jax_to_torch(obs, device=self.device), self._state, info

    def step(self, action):
        random_seed = np.random.randint(0, 2**32)
        self.seed(random_seed)
        
        action = brax_torch.torch_to_jax(action)
        self._state, obs, reward, done, info = self._step(self._key, self._state, action)
        obs = brax_torch.jax_to_torch(obs, device=self.device)
        reward = brax_torch.jax_to_torch(reward, device=self.device)
        terminateds = brax_torch.jax_to_torch(done, device=self.device)
        truncateds = brax_torch.jax_to_torch(done, device=self.device)
        info = brax_torch.jax_to_torch(info, device=self.device)
        return obs, reward, terminateds, truncateds, self._state, info 

    def seed(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)

class CraftaxRecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None
        self.state = None

    def reset(self, **kwargs):
        observations, state, info = super().reset(**kwargs)
        self.state = state
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations, info

    def step(self, action):
        observations, rewards, terms, truncs, state, infos = super().step(action)
        self.state = state
        self.episode_returns += rewards.cpu().numpy()
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - terms.cpu().numpy()
        self.episode_lengths *= 1 - terms.cpu().numpy().astype(np.int32)
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            terms,
            truncs,
            infos,
        )

def make_craftax_env(
    env_id: str = "Craftax",
    num_envs: int = 32,
    reset_ratio: int = 8,
    device: str = "cpu",
):
    if env_id == "Craftax-Classic":
        env = CraftaxClassicSymbolicEnv()
    elif env_id == "Craftax":
        env = CraftaxSymbolicEnv()
    else:
        raise ValueError(f"Unknown environment: {env_id}")

    env = LogWrapper(env)
    env = OptimisticResetVecEnvWrapper(env, num_envs=num_envs, reset_ratio=reset_ratio)
    env = TorchWrapper(env, device=device)
    env = CraftaxRecordEpisodeStatistics(env)
    env.num_envs = num_envs
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    return env

def make_losses_csv(save_path="./"):
    file = os.path.join(save_path, 'losses.csv')
    with open(file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Timestamp', 'SPS', 'Step', 'Learning Rate', 'Value Loss', 'Policy Loss', 'Entropy',
                                'Old Approx KL', 'Approx KL', 'Clipfrac', 'Explained Variance'])
    return file

def make_training_csv_craftax_classic(save_path="./"):
    file = os.path.join(save_path, 'training.csv')
    with open(file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Timestamp', 'SPS', 'Step', 'Episodic Return', 'Episodic Length', 'collect_coal',
                                'collect_diamond', 'collect_drink', 'collect_iron',
                                'collect_sapling', 'collect_stone', 'collect_wood',
                                'defeat_skeleton', 'defeat_zombie', 'eat_cow',
                                'eat_plant', 'make_iron_pickaxe', 'make_iron_sword',
                                'make_stone_pickaxe', 'make_stone_sword', 'make_wood_pickaxe',
                                'make_wood_sword', 'place_furnace', 'place_plant',
                                'place_stone', 'place_table', 'wake_up'])
    return file

def make_training_csv_craftax(save_path="./"):
    file = os.path.join(save_path, 'training.csv')
    with open(file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Timestamp', 'SPS', 'Step', 'Episodic Return', 'Episodic Length', 'cast_fireball', 'cast_iceball',
                             'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron',
                             'collect_ruby', 'collect_sapling', 'collect_sapphire', 'collect_stone',
                             'collect_wood', 'damage_necromancer', 'defeat_archer', 'defeat_deep_thing',
                             'defeat_fire_elemental', 'defeat_frost_troll', 'defeat_gnome_archer', 'defeat_gnome_warrior',
                             'defeat_ice_elemental', 'defeat_knight', 'defeat_kobold', 'defeat_lizard', 'defeat_necromancer',
                             'defeat_orc_mage', 'defeat_orc_solider', 'defeat_pigman', 'defeat_skeleton', 'defeat_troll',
                             'defeat_zombie', 'drink_potion', 'eat_bat', 'eat_cow', 'eat_plant', 'eat_snail',
                             'enchant_armour', 'enchant_sword', 'enter_dungeon', 'enter_fire_realm', 'enter_gnomish_mines',
                             'enter_graveyard', 'enter_ice_realm', 'enter_sewers', 'enter_troll_mines', 'enter_vault',
                             'find_bow', 'fire_bow', 'learn_fireball', 'learn_iceball', 'make_arrow',
                             'make_diamond_armour', 'make_diamond_pickaxe', 'make_diamond_sword', 'make_iron_armour',
                             'make_iron_pickaxe', 'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
                             'make_torch', 'make_wood_pickaxe', 'make_wood_sword', 'open_chest', 'place_furnace',
                             'place_plant', 'place_stone', 'place_table', 'place_torch', 'wake_up'])
    return file
    
def write_row_csv(file, row):
    with open(file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(row)