# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.gat_layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_heads)
        ])
        self.attn_fc = nn.ModuleList([
            nn.Linear(2 * out_dim, 1, bias=False) for _ in range(num_heads)
        ])

    def forward(self, x, edge_index):
        """
        x: [batch_size, in_dim]
        edge_index: [2, num_edges]
        """
        batch_size = x.size(0)
        out = []
        for head in range(self.num_heads):
            h = self.gat_layers[head](x)  # [batch_size, out_dim]
            # Compute attention scores
            src, dst = edge_index  # [num_edges]
            # Ensure src and dst are valid indices
            if torch.any(src >= batch_size) or torch.any(dst >= batch_size) or torch.any(src < 0) or torch.any(dst < 0):
                raise ValueError("edge_index contains invalid node indices.")
            h_src = h[src]  # [num_edges, out_dim]
            h_dst = h[dst]  # [num_edges, out_dim]
            attn_input = torch.cat([h_src, h_dst], dim=1)  # [num_edges, 2*out_dim]
            e = F.leaky_relu(self.attn_fc[head](attn_input)).squeeze(1)  # [num_edges]
            # Compute softmax for attention
            alpha = torch.softmax(e, dim=0)  # [num_edges]
            # Multiply by attention coefficients
            h_src_alpha = h_src * alpha.unsqueeze(1)  # [num_edges, out_dim]
            # Aggregate messages
            out_head = torch.zeros_like(h)  # [batch_size, out_dim]
            out_head = out_head.scatter_add_(0, dst.unsqueeze(1).expand(-1, h_src_alpha.size(1)), h_src_alpha)
            out.append(out_head)
        # Concatenate heads
        out = torch.cat(out, dim=1)  # [batch_size, out_dim * num_heads]
        return out

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, concat=True):
        super(MultiHeadGATLayer, self).__init__()
        self.gat = GATLayer(in_dim, out_dim, num_heads)
        self.concat = concat

    def forward(self, x, edge_index):
        out = self.gat(x, edge_index)  # [batch_size, out_dim * num_heads]
        if self.concat:
            return out  # Concatenated heads
        else:
            return out.mean(dim=1)  # Averaged heads

class STAN(nn.Module):
    def __init__(self, g, in_dim, hidden_dim1, hidden_dim2, gru_dim, num_heads, pred_window, device):
        super(STAN, self).__init__()
        self.g = g
        self.device = device
        # Define GAT layers
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim1, num_heads, concat=True)
        self.layer2 = MultiHeadGATLayer(hidden_dim1 * num_heads, hidden_dim2, num_heads, concat=True)
        # Define GRU
        self.gru = nn.GRU(hidden_dim2 * num_heads, gru_dim, batch_first=True)
        # Define prediction layers
        self.nn_res_I = nn.Linear(gru_dim + 2, pred_window)
        self.nn_res_R = nn.Linear(gru_dim + 2, pred_window)
        self.nn_res_sir = nn.Linear(gru_dim + 2, 2)  # Predict alpha and beta

    def forward(self, dynamic, cI, cR, N, I, R, h=None):
        """
        dynamic: [time, batch, features]
        cI: [time, batch]
        cR: [time, batch]
        N: [batch, 1]
        I: [batch, 1]
        R: [batch, 1]
        h: [num_layers, batch, gru_dim]
        """
        for each_step in range(dynamic.size(0)):
            x = dynamic[each_step]  # [batch, features]
            x = self.layer1(x, self.g.edge_index)  # [batch, hidden_dim1 * num_heads]
            x = F.elu(x)
            x = self.layer2(x, self.g.edge_index)  # [batch, hidden_dim2 * num_heads]
            x = F.elu(x)
            # Aggregate node features, here mean
            cur_h = torch.mean(x, dim=1, keepdim=True)  # [batch, 1, hidden_dim2 * num_heads]

            if h is None:
                h, _ = self.gru(cur_h)  # h: [batch, 1, gru_dim]
            else:
                h, _ = self.gru(cur_h, h)  # h: [batch, 1, gru_dim]

            # Reshape cI and cR to [batch, 1, 1]
            cI_step = cI[each_step].unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
            cR_step = cR[each_step].unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]

            # Concatenate along last dimension
            hc = torch.cat((h, cI_step, cR_step), dim=2)  # [batch, 1, gru_dim + 2]

            # Make predictions
            pred_I = self.nn_res_I(hc)  # [batch, 1, pred_window]
            pred_R = self.nn_res_R(hc)  # [batch, 1, pred_window]

            # Predict SIR parameters
            sir_params = self.nn_res_sir(hc).sigmoid()  # [batch, 1, 2]
            alpha, beta = sir_params.split(1, dim=2)  # Each: [batch, 1, 1]

            # Calculate S using S = N - I - R
            S = torch.clamp(N - I - R, min=0)  # [batch, 1]

            # Compute phylogenetic SIR changes
            phy_I = alpha * I * (S / N) - beta * I  # [batch, 1, 1]
            phy_R = beta * I  # [batch, 1, 1]

        return pred_I, pred_R, phy_I, phy_R, h
