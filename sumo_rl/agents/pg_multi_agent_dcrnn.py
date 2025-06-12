from ..models.dcrnn_model import *
import torch_geometric

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
class PGMultiAgent:
    def __init__(self, k, hops, model_args, device, gamma=0.99, lr=1e-4):
        # global graph structure
        self.ts_indx = model_args['ts_indx'] # global
        self.adj_list = model_args['adj_list'] # np.array: [|E|, 2] global
        self.feat_dim = model_args['feat_dim']
        self.max_diffusion_step = model_args['max_diffusion_step']
        self.max_green_phases = model_args['max_green_phases']
        self.hid_dim = model_args['hid_dim']
        self.num_nodes = model_args['num_nodes']
        self.num_rnn_layers = model_args['num_rnn_layers']
        self.filter_type = model_args['filter_type']
        self.mask = model_args['mask']  # global
        self.edge_index = torch.tensor(self.adj_list.T, dtype=torch.long)  # tensor: [2, |E|] global

        self.models = {}
        self.optimizers = {}
        self.device = device
        self.gamma = gamma
        self.lr = lr
        self.k = k
        self.hops = hops
        self.model_args = model_args
        #self.model_type = model_type
        self.last_k_observations = {ts: [] for ts in self.ts_indx.keys()}
        self.no_op = {ts: 0 for ts in self.ts_indx.keys()}

        for ts_id in self.ts_indx.keys():
            subgraph_nodes, subgraph_edge_index = self.create_local_graph(ts_id)  # subgraph
            # subgraph_nodes: node_idx included in the subgraph
            # subgraph_edge_index: tensor [2, |subE|]
            ts_idx = self.ts_indx[ts_id]
            # construct local binary adjacency matrix from subgraph np.array [2, |subE|]
            adj_mx = self.construct_binary_adj_mat(subgraph_edge_index, num_nodes=subgraph_nodes.size(0))
            #print("adj_mx", adj_mx)

            encoder = DCRNNEncoder(
                input_dim=self.feat_dim,  # Density and queue
                adj_mat=adj_mx,
                max_diffusion_step=self.max_diffusion_step,
                hid_dim=self.hid_dim,
                num_nodes=subgraph_nodes.size(0),
                num_rnn_layers=self.num_rnn_layers,
                filter_type="dual_random_walk"
            ).to(self.device)

            head = SingleTLPhasePredictor(
                hid_dim=self.hid_dim,
                input_dim=self.feat_dim,
                max_green_phases=self.max_green_phases,
                mask=self.mask  # global
            ).to(self.device)

            model = TSModel(encoder, head).to(self.device)
            self.models[ts_id] = model

        for ts_id in self.ts_indx.keys():
            self.optimizers[ts_id] = optim.Adam(self.models[ts_id].parameters(), lr=lr)

    def update(self, new_obs):
        for ts in self.last_k_observations.keys():
            self.last_k_observations[ts].append(new_obs[ts])
            if len(self.last_k_observations[ts]) > self.k:
                self.last_k_observations[ts].pop(0)

    def construct_binary_adj_mat(self, subgraph_edge_index, num_nodes):
        """
        Construct a binary adjacency matrix from subgraph edge indices.

        Args:
            subgraph_edge_index (torch.Tensor): Tensor of shape [2, |subE|], representing edges in the subgraph.
            num_nodes (int): Total number of nodes in the subgraph.

        Returns:
            np.array: Binary adjacency matrix of shape (num_nodes, num_nodes).
        """
        # Initialize an adjacency matrix with zeros
        adj_mat = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        # Iterate through edges and populate the adjacency matrix
        for u, v in subgraph_edge_index.T.cpu().numpy():
            adj_mat[u, v] = 1.0

        adj_mat += np.eye(adj_mat.shape[0])

        return adj_mat

    def create_local_graph(self, ts_id, max_lane=None):
        """
        Create a local graph for the given traffic signal, considering both directions.

        Args:
            ts_id (str): Traffic signal ID.
            max_lane (int): Maximum number of lanes to pad density/queue.

        Returns:
            (subgraph_features (torch.Tensor): Features for the local subgraph nodes if max_lane is not None).
            subgraph_nodes (torch.Tensor): Indices of nodes in the subgraph.
            subgraph_edge_index (torch.Tensor): Edge index of the subgraph. [2, |subE|]
        """
        node_index = self.ts_indx[ts_id]

        st_nodes, st_edge_index, _, _ = torch_geometric.utils.k_hop_subgraph(
            node_idx=node_index, num_hops=self.hops, edge_index=self.edge_index, relabel_nodes=False, flow="source_to_target"
        )

        ts_nodes, ts_edge_index, _, _ = torch_geometric.utils.k_hop_subgraph(
            node_idx=node_index, num_hops=self.hops, edge_index=self.edge_index, relabel_nodes=False, flow="target_to_source"
        )

        subgraph_nodes = torch.cat((st_nodes, ts_nodes)).unique(sorted=False)
        subgraph_edge_index = torch.cat((st_edge_index, ts_edge_index), dim=1).unique(dim=1)

        nodes_mapping = {node.item(): i for i, node in enumerate(subgraph_nodes)}

        # Map edges using the node mapping
        subgraph_edge_index = torch.stack([
            torch.tensor([nodes_mapping[u.item()] for u in subgraph_edge_index[0]]),
            torch.tensor([nodes_mapping[v.item()] for v in subgraph_edge_index[1]])
        ], dim=0)

        # Aggregate features for the combined subgraph
        idx_to_ts_id = {v: k for k, v in self.ts_indx.items()}

        if max_lane is None:
            return subgraph_nodes, subgraph_edge_index

        # create placeholder for subgraph_features: shape (num_timesteps, num_nodes, feature_size)
        subgraph_features = torch.zeros((self.k, len(subgraph_nodes), 2*max_lane), dtype=torch.float32)

        for local_idx, node_idx in enumerate(subgraph_nodes):
            ts_id = idx_to_ts_id.get(node_idx.item(), None)
            if ts_id is None:
                continue
            node_obs = self.last_k_observations[ts_id]

            for t, obs in enumerate(node_obs):
                # Pad density and queue features
                density = torch.tensor(obs["density"], dtype=torch.float32)
                queue = torch.tensor(obs["queue"], dtype=torch.float32)

                padded_density = torch.nn.functional.pad(density, (0, max_lane - len(density)))
                padded_queue = torch.nn.functional.pad(queue, (0, max_lane - len(queue)))

                # Concatenate density and queue features
                subgraph_features[t, local_idx, :] = torch.cat((padded_density, padded_queue))

        return subgraph_features, subgraph_nodes, subgraph_edge_index

    def train(self, env, num_episodes, model_dir):
        # keep track of the last k observations
        # for num_episodes:
        # env.reset()
        # enter while loop until truncation:
        # 1. if warmup phase, get the new observation shape {ts_id: observation_space} and update the last_k_obs, then continue
        # 2. at step k, set fix_ts = False and start RL control
        # 3. each agent build their local graph based on current last_k_obs
        # 3. output actions (calculate its logprob)
        # 4. env.step(action) = observations, reward, termination, truncation, info
        # 5. update the last_k_observations, record reward
        # outside the while loop, look at the rewards and log_prob, update policy using PG
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}")
            obs, _ = env.reset()
            agents = env.agents
            self.update(obs)  # initial observation
            agent_experiences = {agent_name: {'log_probs': [], 'rewards': []} for agent_name in env.agents}
            sumo_env = env.aec_env.env.env.env
            sumo_env.fixed_ts = True  # warmup phase
            max_lanes = max(len(ts.lanes) for ts in sumo_env.traffic_signals.values())
            done = False
            it = 0
            while not done:
                # Warmup phase with default traffic light control
                if it < self.k:
                    obs, _, _, _, _ = env.step(self.no_op)  # Run default traffic light control
                    self.update(obs)
                    it += 1
                    continue

                # RL control starts after warmup
                if it == self.k:
                    sumo_env.fixed_ts = False  # Switch to RL control
                    for _, ts in sumo_env.traffic_signals.items():
                        ts.run_rl_agents()  # Activate RL agents

                it += 1
                actions = {}
                for agent_name in env.agents:
                    agent_idx = self.ts_indx[agent_name]
                    # Create local graph
                    agents_features, subgraph_nodes, subgraph_edge_index = self.create_local_graph(
                        agent_name, max_lanes
                    )
                    agents_features = agents_features.unsqueeze(1).to(self.device)
                    subgraph_nodes = subgraph_nodes.to(self.device)
                    initial_hidden_state = torch.zeros((self.num_rnn_layers, 1, subgraph_nodes.size(0) * self.hid_dim), device=self.device)

                    model = self.models[agent_name]
                    logits = model(agents_features, initial_hidden_state, agent_idx, subgraph_nodes).squeeze(0)

                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    actions[agent_name] = action.item()
                    log_prob = dist.log_prob(action)
                    agent_experiences[agent_name]['log_probs'].append(log_prob)

                # Step the environment
                observations, rewards, terminations, truncations, infos = env.step(actions)
                self.update(obs)

                # Store rewards
                for agent_name in env.agents:
                    reward = rewards[agent_name]
                    agent_experiences[agent_name]['rewards'].append(reward)

                # Check if all agents are done
                done = all(terminations.values()) or all(truncations.values())

            # At the end of the episode, update the policy for each agent
            for agent_name in agents:
                print("Agent {} finished after {} timesteps".format(agent_name, it))
                model = self.models[agent_name]
                optimizer = self.optimizers[agent_name]
                optimizer.zero_grad()
                log_probs = agent_experiences[agent_name]['log_probs']
                rewards = agent_experiences[agent_name]['rewards']

                # Compute returns (discounted rewards)
                returns = []
                Gt = 0
                for r in reversed(rewards):
                    Gt = r + self.gamma * Gt
                    returns.insert(0, Gt)
                returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
                # Normalize returns
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

                # Compute policy loss
                policy_loss = 0  # Scalar accumulation
                for log_prob, Gt in zip(log_probs, returns):
                    policy_loss += -log_prob * Gt
                # Backpropagate the policy loss
                policy_loss.backward()
                optimizer.step()
                print(f"Episode {episode + 1}, Agent {agent_name}, Total Reward: {sum(rewards)}, Loss: {policy_loss.item()}")

        base_env = env.unwrapped.env
        base_env.save_csv(base_env.out_csv_name, num_episodes)
        # Save models after training
        self.save_model(model_dir)

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for agent_name, model in self.models.items():
            model_path = os.path.join(output_dir, f"{agent_name}_model_dcrnn.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved model for {agent_name} to {model_path}")
