from stable_baselines3.common.env_checker import check_env
from nudging_env import NudgingEnv


env = NudgingEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)