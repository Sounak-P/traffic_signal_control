import numpy as np
from .util import *
import torch
import torch.nn as nn
from .base_model import BaseModel

class DiffusionGraphConv(BaseModel):
    def __init__(self, supports, input_dim, hid_dim, num_nodes, max_diffusion_step, output_dim, bias_start=0.0):
        super(DiffusionGraphConv, self).__init__()
        self.num_matrices = len(supports) * max_diffusion_step + 1  # Don't forget to add for x itself.
        input_size = input_dim + hid_dim
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._supports = supports
        self.weight = nn.Parameter(torch.FloatTensor(size=(input_size*self.num_matrices, output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self, inputs, state):
        """
        Diffusion Graph convolution with graph matrix
        :param inputs: (bs, nodes * input_dim)
        :param state: (bs, nodes * hid_dim)
        :return:
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2) # [X^(t), H^(t-1)] (bs, nodes, input_dim + hid_dim)
        input_size = inputs_and_state.shape[2]

        x = inputs_and_state

        x = x.permute(1, 2, 0).reshape(self._num_nodes, -1)  # Shape: (num_nodes, input_size * bs)

        res = [x]

        if self._max_diffusion_step > 0:
            # forward and reverse diffusion matrices
            for support in self._supports:
                # Ensure support matrix is on the same device as x
                device = x.device
                if support.device != device:
                    support = support.to(device)
                x_k = x
                for k in range(self._max_diffusion_step):
                    x_k = torch.sparse.mm(support, x_k)
                    res.append(x_k)

        x = torch.stack(res, dim=0)  # Shape: (num_matrices, nodes, input_size * bs)

        x = x.view(self.num_matrices, self._num_nodes, input_size, batch_size).permute(3, 1, 2, 0) # (bs, nodes, input_size, num_matrices)
        x = x.reshape(batch_size * self._num_nodes, input_size * self.num_matrices)

        x = torch.matmul(x, self.weight)  # (bs, nodes, output_dim)
        x = x + self.biases

        output = x.view(batch_size, -1)  # (bs, nodes * output_dim)
        return output


class DCGRUCell(BaseModel):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """
    def __init__(self, input_dim, hid_dim, adj_mat, max_diffusion_step, num_nodes,
                 num_proj=None, activation=torch.tanh, use_gc_for_ru=True, filter_type='laplacian'):
        """
        :param hid_dim: the hidden dim of rnn
        :param adj_mat: the (weighted) adjacency matrix of the graph, in numpy ndarray form e.g. w_ij = exp(-dist(vi,vj)^2/sigma^2
        :param max_diffusion_step: the max diffusion step
        :param num_nodes:
        :param num_proj: num of output dim
        :param activation: if None, don't do activation for cell state
        :param use_gc_for_ru: decide whether to use graph convolution inside rnn
        :param filter_type: "laplacian", "random_walk", "dual_random_walk"
        """
        super(DCGRUCell, self).__init__()
        self._activation = activation
        self._num_nodes = num_nodes
        self._hid_dim = hid_dim
        self._max_diffusion_step = max_diffusion_step
        self._num_proj = num_proj
        self._use_gc_for_ru = use_gc_for_ru
        self._supports = []
        supports = []
        if filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mat, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(calculate_random_walk_matrix(adj_mat).T)
        elif filter_type == "dual_random_walk":
            supports.append(calculate_random_walk_matrix(adj_mat))
            supports.append(calculate_random_walk_matrix(adj_mat.T))
        else:
            supports.append(calculate_scaled_laplacian(adj_mat))

        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))  # to PyTorch sparse tensor

        self.dconv_gate = DiffusionGraphConv(supports=self._supports, input_dim=input_dim,
                                             hid_dim=hid_dim, num_nodes=num_nodes,
                                             max_diffusion_step=max_diffusion_step,
                                             output_dim=hid_dim*2)

        self.dconv_candidate = DiffusionGraphConv(supports=self._supports, input_dim=input_dim,
                                                  hid_dim=hid_dim, num_nodes=num_nodes,
                                                  max_diffusion_step=max_diffusion_step,
                                                  output_dim=hid_dim)

        if not use_gc_for_ru:
            self._fc = nn.Linear(input_dim + hid_dim, 2 * hid_dim)

        if num_proj is not None:
            self.project = nn.Linear(self._hid_dim, self._num_proj)

    def to(self, device):
        """Override to() method to handle sparse matrices device placement"""
        # Move the module parameters and buffers to the device
        super(DCGRUCell, self).to(device)
        
        # Move sparse support matrices to the device
        self._supports = [support.to(device) for support in self._supports]
        
        # Update the DiffusionGraphConv layers with moved supports
        self.dconv_gate._supports = self._supports
        self.dconv_candidate._supports = self._supports
        
        return self

    def forward(self, inputs, state):
        """
        :param inputs: (B, num_nodes * input_dim)
        :param state: (B, num_nodes * hid_dim)
        :return:
        """
        fn = self.dconv_gate if self._use_gc_for_ru else self._fc

        gate_output_size = 2 * self._hid_dim
        gates = torch.sigmoid(fn(inputs, state)) # (bs, nodes * gate_output_size)
        gates = gates.view(-1, self._num_nodes, gate_output_size)
        #print("gate_output_size ", gate_output_size)
        # Split the gates tensor into reset (r) and update (u) gates
        r, u = torch.split(gates, int(gate_output_size/2), dim=-1)
        #print("r shape: ", r.shape)
        #print(f"num_nodes {self._num_nodes} and hid_dim {self._hid_dim}")
        r = r.reshape(-1, self._num_nodes * self._hid_dim)
        u = u.reshape(-1, self._num_nodes * self._hid_dim)

        c = self.dconv_candidate(inputs, r * state)  # (bs, nodes * hid_dim)
        if self._activation is not None:
            c = self._activation(c)
        output = new_state = u * state + (1 - u) * c
        if self._num_proj is not None:
            batch_size = inputs.shape[0]
            output = self.project(new_state.view(-1, self._hid_dim)).view(batch_size, self._num_nodes * self._num_proj)

        return output, new_state

    @staticmethod
    def _build_sparse_matrix(L):
        """
        build pytorch sparse tensor from scipy sparse matrix
        reference: https://stackoverflow.com/questions/50665141
        :return:
        """
        shape = L.shape
        i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        v = torch.FloatTensor(L.data)
        return torch.sparse_coo_tensor(i, v, torch.Size(shape))
