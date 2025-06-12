import torch
import torch.nn as nn
from .dcrnn_cell import DCGRUCell
from .base_model import BaseModel

class DCRNNEncoder(BaseModel):
    def __init__(self, input_dim, adj_mat, max_diffusion_step, hid_dim, num_nodes,
                 num_rnn_layers, filter_type):
        super(DCRNNEncoder, self).__init__()
        self._hid_dim = hid_dim
        self._num_rnn_layers = num_rnn_layers
        self._num_nodes = num_nodes

        encoding_cells = list()

        # the first layer has different input_dim
        encoding_cells.append(DCGRUCell(input_dim=input_dim, hid_dim=hid_dim, adj_mat=adj_mat,
                                        max_diffusion_step=max_diffusion_step,
                                        num_nodes=num_nodes, filter_type=filter_type))

        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(DCGRUCell(input_dim=hid_dim, hid_dim=hid_dim, adj_mat=adj_mat,
                                            max_diffusion_step=max_diffusion_step,
                                            num_nodes=num_nodes, filter_type=filter_type))
        self.encoding_cells = nn.ModuleList(encoding_cells)

    def forward(self, inputs, initial_hidden_state):
        # inputs shape: (seq_len, batch, num_nodes, input_dim)
        # inputs to cell: (batch, num_nodes * input_dim)
        # init_hidden_state: (num_layers, batch, num_nodes * hid_dim)
        seq_length, batch_size = inputs.shape[:2]
        inputs = inputs.view(seq_length, batch_size, -1)  # (seq_len, batch, num_nodes * input_dim)

        current_inputs = inputs
        #final_hidden_states = []

        for i in range(self._num_rnn_layers):
            hidden_state = initial_hidden_state[i]
            layer_outputs = []

            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i](current_inputs[t], hidden_state)
                layer_outputs.append(hidden_state)

            #final_hidden_states.append(hidden_state)

            # Update current_inputs for the next layer
            current_inputs = torch.stack(layer_outputs,
                                         dim=0)  # Shape: (seq_length, batch_size, num_nodes * hid_dim)

        # output_hidden = torch.stack(final_hidden_states, dim=0)  # (num_layers, batch_size, num_nodes * hid_dim)
        # return output_hidden, current_inputs

        final_output = current_inputs[-1].view(batch_size, self._num_nodes,
                                               self._hid_dim)  # Shape: (batch_size, num_nodes, hid_dim)
        return final_output


class SingleTLPhasePredictor(nn.Module):
    def __init__(self, hid_dim, input_dim, max_green_phases, mask):
        """
        Predicts actions for traffic nodes based on the hidden state from the encoder and local features.

        Args:
            hid_dim (int): Hidden dimension size from the DCRNN encoder.
            input_dim (int): Input dimension size from the DCRNN encoder.
            max_green_phases (int): Maximum number of possible green phases for any traffic node.
            mask (torch.Tensor): A binary mask of shape (num_nodes, max_green_phases)
        """
        super(SingleTLPhasePredictor, self).__init__()
        self.hid_dim = hid_dim
        # Register mask as a buffer so it gets moved to the correct device with the model
        self.register_buffer('mask', mask)
        self.input_dim = hid_dim + input_dim

        # MLP to map concatenated features to action logits
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, max_green_phases)
        )

    def forward(self, hidden_states, input_features, agent_idx, subgraph_nodes):  # agent_idx: global index
        # hidden_states: (bs, subgraph_nodes, hid_dim)
        # input_features:  (bs, subgraph_nodes, input_dim)
        local_agent_idx = (subgraph_nodes == agent_idx).nonzero(as_tuple=True)[0].item()
        x = torch.cat([hidden_states[:, local_agent_idx], input_features[:, local_agent_idx]], dim=1)
        logits = self.mlp(x)
        logits = logits + (1 - self.mask[agent_idx]) * -1e9
        return logits



class TLPhasePredictor(nn.Module):
    def __init__(self, hid_dim, input_dim, num_nodes, num_virtual_nodes, max_green_phases, mask):
        """
        Predicts actions for traffic nodes based on the hidden state from the encoder and local features.

        Args:
            hid_dim (int): Hidden dimension size from the DCRNN encoder.
            input_dim (int): Input dimension size from the DCRNN encoder.
            num_nodes (int):
            num_virtual_nodes (int): default = 2
            max_green_phases (int): Maximum number of possible green phases for any traffic node.
            mask (torch.Tensor): A binary mask of shape (num_nodes, max_green_phases)
        """
        super(TLPhasePredictor, self).__init__()
        self.hid_dim = hid_dim
        self.num_ts = num_nodes - num_virtual_nodes  # Exclude virtual incoming/ongoing nodes
        self.num_virtual_nodes = num_virtual_nodes
        self.max_green_phases = max_green_phases
        # Register mask as a buffer so it gets moved to the correct device with the model
        self.register_buffer('mask', mask[:-num_virtual_nodes])

        self.input_dim = hid_dim + input_dim

        # MLP to map concatenated features to action logits
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, max_green_phases)
        )

    def forward(self, hidden_states, input_features):
        """
        Predict actions for traffic nodes.

        Args:
            hidden_states (torch.Tensor): Hidden states of shape (bs, num_nodes, hid_dim).
            input_features (torch.Tensor): Input features of shape (bs, num_nodes, input_dim).

        Returns:
            masked_actions (torch.Tensor): Masked action logits of shape (bs, num_ts, max_green_phases).
        """
        ts_hid_states = hidden_states[:, :-self.num_virtual_nodes, :]  # Shape: (bs, num_ts, hid_dim)
        ts_input_states = input_features[:, :-self.num_virtual_nodes, :]

        ts_input = torch.cat([ts_hid_states, ts_input_states], dim=-1)  # Shape: (bs, num_ts, input_dim)

        # Apply MLP to predict action logits for each node
        action_logits = self.mlp(ts_input)  # Shape: (bs, num_ts, max_green_phases)

        # Apply mask to enforce valid action spaces
        masked_actions = action_logits + (1 - self.mask.unsqueeze(0)) * -1e9

        return masked_actions


class TSModel(nn.Module):
    def __init__(self, encoder, head):
        super(TSModel, self).__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, obs, initial_hidden_state, ts_idx=None, subgraph_nodes=None):  # obs: shape of [num_timesteps, bs, num_nodes, feature_size]
        output_features = self.encoder(obs, initial_hidden_state)  # (bs, num_nodes, hid_dim)

        input_features = obs[-1]  # (bs, num_nodes, input_dim)

        if ts_idx is None:
            logits = self.head(output_features, input_features)  # (bs, num_ts, max_green_phases)
        else:
            logits = self.head(output_features, input_features, ts_idx, subgraph_nodes)
        #if ts_idx is not None:
        #    logits = logits[ts_idx].squeeze(0)

        return logits



