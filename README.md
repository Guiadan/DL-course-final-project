# DL-course-final-project
This repository is a project for the DL course in the Technion

# Requirements
Python 2.7

gym

tqdm

OpenCV2 or Scipy

TensorFlow 0.12.0

PyGame-Learning-Environment - https://github.com/ntasfi/PyGame-Learning-Environment

# Installation
clone this repository
in the directory where PyGame-Learning-Environment is installed copy the following files:
multiagentpuckworld.py
multiagentpuckworld.pyc
puckworld.py
puckworld.pyc
to $PATH/PyGame-Learning-Environment/ple/games/
replace existing files

# reproducing results:
code for reproduction of target update decay:
python main.py --network_header_type=nature --env_name=dqn_agent_target_updates_decay_new --target_q_update_freq_decay=1.1 --use_gpu=True --ple=True --gpu_fraction 1/2 --display=True --ple_game_name='PuckWorld' --memory_size=1 --network_output_type='normal' --t_ep_end=200

code for reproduction of average dqn:
python main.py --network_header_type=nature --env_name=average_dqn_10 --use_gpu=True --ple=True --gpu_fraction 1/2 --display=True --ple_game_name='PuckWorld' --memory_size=1 --network_output_type='normal' --t_ep_end=200 --average_dqn=True --GPU_to_use='1'

code for reproduction of combination:
python main.py --network_header_type=nature --env_name=average_dqn_10_and_target_update_decay_new --use_gpu=True --ple=True --gpu_fraction 2/3 --display=True --ple_game_name='PuckWorld' --memory_size=10 --network_output_type='normal' --t_ep_end=2000 --average_dqn=True --average_dqn=True --target_q_update_freq_decay=1.1 --GPU_to_use='0' --t_target_q_update_freq=10

code for reproduction of multi agent experiment:
python main.py --network_header_type=nature --env_name=multi_agent_target_decay --target_q_update_freq_decay=1.1 --use_gpu=True --ple=True --gpu_fraction 4/5 --display=True --memory_size=10 --ple_agents=2 --network_output_type='normal' --t_ep_end=4000 --GPU_to_use='0'
