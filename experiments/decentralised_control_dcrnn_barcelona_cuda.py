import os
import sys
import gc
import torch

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

import gymnasium as gym
import sumo_rl
from sumo_rl.models.util import *
from sumo_rl.agents.pg_multi_agent_dcrnn import PGMultiAgent
from sumo_rl.models.dcrnn_model import *
from sumo_rl.models.transformer_model import PolicyNetwork

import torch.optim as optim

# GPU Memory optimization settings
def setup_gpu_optimization():
    if torch.cuda.is_available():
        # Set memory fraction to use maximum available memory
        torch.cuda.set_per_process_memory_fraction(0.95)
        # Enable memory efficient attention if available
        torch.backends.cuda.enable_flash_sdp(True)
        # Clear cache before starting
        torch.cuda.empty_cache()
        # Enable memory pooling
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Use absolute paths
NET_FILE = os.path.join(PROJECT_ROOT, 'sumo_rl', 'nets', 'Barcelona', 'static_mid-area.net.xml')
ROUTE_FILE = os.path.join(PROJECT_ROOT, 'sumo_rl', 'nets', 'Barcelona', 'mid-area.rou.xml')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'results', 'my_result_dcrnn_barcelona')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# Device optimization
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_MIXED_PRECISION = torch.cuda.is_available()  # Enable mixed precision on GPU
USE_GRADIENT_CHECKPOINTING = torch.cuda.is_available()  # Enable gradient checkpointing on GPU

if DEVICE == 'cuda':
    setup_gpu_optimization()

print(f"Using device: {DEVICE}")
if USE_MIXED_PRECISION:
    print("Mixed precision training: Enabled")

# Environment setup with memory optimization
train_env = sumo_rl.parallel_env(
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    out_csv_name=OUTPUT_CSV,
    use_gui=False,
    num_seconds=3000,
    begin_time=100,
    fixed_ts=True,
    reward_fn="weighted_wait_queue"
)

# Build graph representation
traffic_signals = [ts for _, ts in train_env.aec_env.env.env.env.traffic_signals.items()]
max_lanes = max(len(ts.lanes) for ts in traffic_signals)
max_green_phases = max(ts.num_green_phases for ts in traffic_signals)
ts_phases = [ts.num_green_phases for ts in traffic_signals]
feature_size = 2*max_lanes

# Optimized hyperparameters for V100
if DEVICE == 'cuda':
    # Larger parameters for GPU with 36GB memory
    hid_dim = 256  # Increased from 128
    num_rnn_layers = 2  # Increased from 1
    max_diffusion_step = 8  # Increased from 5
else:
    # Conservative parameters for CPU
    hid_dim = 128
    num_rnn_layers = 1
    max_diffusion_step = 5

num_virtual_nodes = 2
filter_type = "dual_random_walk"

ts_indx, num_nodes, lanes_index, adj_list, incoming_lane_ts, outgoing_lane_ts = construct_graph_representation(traffic_signals)
action_mask = create_action_mask(num_nodes, max_green_phases, ts_phases)

k = 5
hops = 2

# DCRNN model arguments
model_args = {
    "ts_indx": ts_indx,
    "adj_list": adj_list,
    "num_nodes": num_nodes,
    "max_diffusion_step": max_diffusion_step,
    "max_green_phases": max_green_phases,
    "feat_dim": feature_size,
    "output_features": max_green_phases,
    "hid_dim": hid_dim,
    "mask": action_mask,
    "num_rnn_layers": num_rnn_layers,
    "filter_type": filter_type
}

# Initialize PGMultiAgent with optimizations
PGMultiAgent = PGMultiAgent(k, hops, model_args, DEVICE)

# Training with GPU memory optimization
if DEVICE == 'cuda':
    num_episodes = 10  # More episodes for GPU training
    print("Starting GPU-optimized training...")
    print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Monitor memory usage during training
    def print_memory_usage():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            cached = torch.cuda.memory_reserved(0) / 1e9
            print(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    
    print_memory_usage()
else:
    num_episodes = 2  # Original episodes for CPU
    print("Starting CPU training...")

PGMultiAgent.train(train_env, num_episodes=num_episodes, model_dir=MODEL_DIR)

if DEVICE == 'cuda':
    print_memory_usage()
    torch.cuda.empty_cache()  # Clean up after training