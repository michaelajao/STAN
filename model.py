import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.gat_layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(self.num_heads)
        ])
        self.attn_fc = nn.ModuleList([
            nn.Linear(2 * out_dim, 1, bias=False) for _ in range(self.num_heads)
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

            src, dst = edge_index
            src = src.to(x.device)
            dst = dst.to(x.device)

            h_src = h[src]
            h_dst = h[dst]
            attn_input = torch.cat([h_src, h_dst], dim=1)
            e = F.leaky_relu(self.attn_fc[head](attn_input)).squeeze(1)

            alpha = torch.softmax(e, dim=0)
            h_src_alpha = h_src * alpha.unsqueeze(1)
            out_head = torch.zeros_like(h)
            out_head = out_head.scatter_add_(0, dst.unsqueeze(1).expand(-1, h_src_alpha.size(1)), h_src_alpha)
            out.append(out_head)

        out = torch.cat(out, dim=1)  # [batch_size, out_dim * num_heads]
        return out

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, concat=True):
        super(MultiHeadGATLayer, self).__init__()
        self.gat = GATLayer(in_dim, out_dim, num_heads)
        self.concat = concat
        self.num_heads = num_heads
        self.out_dim = out_dim

    def forward(self, x, edge_index):
        out = self.gat(x, edge_index)
        if self.concat:
            return out
        else:
            return out.view(out.size(0), self.num_heads, self.out_dim).mean(dim=1)

class STAN(nn.Module):
    def __init__(self, g, in_dim, hidden_dim1, hidden_dim2, gru_dim, num_heads, pred_window, device):
        super(STAN, self).__init__()
        self.g = g
        self.device = device
        self.pred_window = pred_window

        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim1, num_heads, concat=True)
        self.layer2 = MultiHeadGATLayer(hidden_dim1 * num_heads, hidden_dim2, num_heads, concat=True)
        self.gru = nn.GRU(hidden_dim2 * num_heads, gru_dim, batch_first=True)

        self.nn_res_I = nn.Linear(gru_dim + 2, pred_window)
        self.nn_res_R = nn.Linear(gru_dim + 2, pred_window)
        self.nn_res_sir = nn.Linear(gru_dim + 2, 2)  # alpha and beta

    def forward(self, dynamic, cI, cR, N, I, R, h=None):
        """
        dynamic: [time, batch, features]
        cI: [time, batch, 1]
        cR: [time, batch, 1]
        N: [batch, 1]
        I: [time, batch, 1]
        R: [time, batch, 1]
        h: optional hidden state for GRU
        """
        pred_I_list = []
        pred_R_list = []
        phy_I_list = []
        phy_R_list = []

        seq_len = dynamic.size(0)
        batch_size = dynamic.size(1)

        for each_step in range(seq_len):
            x = dynamic[each_step]  # [batch, features]
            x = self.layer1(x, self.g.edge_index)
            x = F.elu(x)
            x = self.layer2(x, self.g.edge_index)
            x = F.elu(x)

            # GRU expects [batch, seq=1, features]
            cur_h = x.unsqueeze(1)  # [batch, 1, hidden_dim]

            if h is None:
                out, h = self.gru(cur_h)  # out: [batch, 1, gru_dim]
            else:
                out, h = self.gru(cur_h, h)

            # cI[each_step]: [batch,1], we want [batch,1,1]
            cI_step = cI[each_step].unsqueeze(1)  # [batch,1,1]
            cR_step = cR[each_step].unsqueeze(1)  # [batch,1,1]

            # Concatenate along dim=2: out: [batch,1,gru_dim], cI_step: [batch,1,1], cR_step: [batch,1,1]
            hc = torch.cat((out, cI_step, cR_step), dim=2)  # [batch, 1, gru_dim+2]

            pred_I = self.nn_res_I(hc)  # [batch, 1, pred_window]
            pred_R = self.nn_res_R(hc)  # [batch, 1, pred_window]

            sir_params = self.nn_res_sir(hc).sigmoid()  # [batch, 1, 2]
            alpha, beta = sir_params.split(1, dim=2)  # [batch, 1, 1]

            I_step = I[each_step]  # [batch,1]
            R_step = R[each_step]  # [batch,1]
            S = torch.clamp(N - I_step - R_step, min=0)  # [batch,1]

            phy_I = alpha * I_step * (S / N) - beta * I_step  # [batch,1,1]
            phy_R = beta * I_step                            # [batch,1,1]

            phy_I = phy_I.squeeze(-1)  # [batch,1]
            phy_R = phy_R.squeeze(-1)  # [batch,1]

            pred_I_list.append(pred_I)   # [batch,1,pred_window]
            pred_R_list.append(pred_R)   # [batch,1,pred_window]
            phy_I_list.append(phy_I)     # [batch,1]
            phy_R_list.append(phy_R)     # [batch,1]

        pred_I_tensor = torch.cat(pred_I_list, dim=1)  # [batch, time, pred_window]
        pred_R_tensor = torch.cat(pred_R_list, dim=1)  # [batch, time, pred_window]

        phy_I_tensor = torch.stack(phy_I_list, dim=0)  # [time, batch, 1]
        phy_R_tensor = torch.stack(phy_R_list, dim=0)  # [time, batch, 1]

        # Expand phy to match pred_window
        phy_I_tensor = phy_I_tensor.expand(-1, -1, self.pred_window)  # [time,batch,pred_window]
        phy_R_tensor = phy_R_tensor.expand(-1, -1, self.pred_window)

        # Permute to [time,batch,pred_window]
        pred_I_tensor = pred_I_tensor.permute(1,0,2)
        pred_R_tensor = pred_R_tensor.permute(1,0,2)
        # phy tensors already [time,batch,pred_window]

        return pred_I_tensor, pred_R_tensor, phy_I_tensor, phy_R_tensor, h
