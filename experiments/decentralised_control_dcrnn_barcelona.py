import os
import sys

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

import gymnasium as gym
import sumo_rl
from sumo_rl.models.util import *
from sumo_rl.agents.pg_multi_agent_dcrnn import PGMultiAgent
from sumo_rl.models.dcrnn_model import *
from sumo_rl.models.transformer_model import PolicyNetwork

import torch
import torch.optim as optim

# Use absolute paths
NET_FILE = os.path.join(PROJECT_ROOT, 'sumo_rl', 'nets', 'Barcelona', 'static_mid-area.net.xml')
ROUTE_FILE = os.path.join(PROJECT_ROOT, 'sumo_rl', 'nets', 'Barcelona', 'mid-area.rou.xml')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'results', 'my_result_dcrnn_barcelona')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

train_env = sumo_rl.parallel_env(
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    out_csv_name=OUTPUT_CSV,
    use_gui=False,
    num_seconds=3000,
    begin_time=100,
    fixed_ts=True,
    reward_fn="weighted_wait_queue"  # r_i,t = queue_i,t + gamma * wait_i,t
)

# Reset environment and get initial observations
# obs, _ = env.reset()

# Build graph representation
traffic_signals = [ts for _, ts in train_env.aec_env.env.env.env.traffic_signals.items()]
max_lanes = max(len(ts.lanes) for ts in traffic_signals)  # max incoming lanes
max_green_phases = max(ts.num_green_phases for ts in traffic_signals)
ts_phases = [ts.num_green_phases for ts in traffic_signals]
feature_size = 2*max_lanes
hid_dim = 128
num_virtual_nodes = 2  # incoming/outgoing
max_diffusion_step = 5
num_rnn_layers = 1
filter_type="dual_random_walk"

ts_indx, num_nodes, lanes_index, adj_list, incoming_lane_ts, outgoing_lane_ts = construct_graph_representation(traffic_signals)
action_mask = create_action_mask(num_nodes, max_green_phases, ts_phases)

k = 5
hops = 2

# dcrnn
model_args = {
    # global graph structure
    "ts_indx": ts_indx,
    "adj_list": adj_list,  # np.array [|E|, 2]
    #"incoming_lane_ts": incoming_lane_ts,
    #"outgoing_lane_ts": outgoing_lane_ts,
    "num_nodes": num_nodes,
    #"num_virtual_nodes": num_virtual_nodes,
    # architecture
    "max_diffusion_step": max_diffusion_step,
    "max_green_phases": max_green_phases,
    "feat_dim": feature_size,
    "output_features": max_green_phases,
    "hid_dim": hid_dim,
    "mask": action_mask,
    "num_rnn_layers": num_rnn_layers,
    "filter_type": filter_type
}

PGMultiAgent = PGMultiAgent(k, hops, model_args, DEVICE)
PGMultiAgent.train(train_env, num_episodes=2, model_dir=MODEL_DIR)

