import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class STAN(nn.Module):
    def __init__(self, g, in_dim, hidden_dim1, hidden_dim2, gru_dim, num_heads, pred_window, device):
        super(STAN, self).__init__()
        self.g = g
        
        # Replace custom GAT layers with PyG's GATConv
        self.conv1 = GATConv(in_dim, hidden_dim1, heads=num_heads, concat=True, dropout=0.6)
        self.conv2 = GATConv(hidden_dim1 * num_heads, hidden_dim2, heads=1, concat=False, dropout=0.6)

        self.pred_window = pred_window
        self.gru = nn.GRUCell(hidden_dim2, gru_dim)
    
        self.nn_res_I = nn.Linear(gru_dim + 2, pred_window)
        self.nn_res_R = nn.Linear(gru_dim + 2, pred_window)

        self.nn_res_sir = nn.Linear(gru_dim + 2, 2)
        
        self.hidden_dim2 = hidden_dim2
        self.gru_dim = gru_dim
        self.device = device

    def forward(self, dynamic, cI, cR, N, I, R, h=None):
        num_loc, timestep, n_feat = dynamic.size()
        N = N.squeeze()

        if h is None:
            h = torch.zeros(1, self.gru_dim).to(self.device)
            gain = nn.init.calculate_gain('relu')
            nn.init.xavier_normal_(h, gain=gain)  

        new_I = []
        new_R = []
        phy_I = []
        phy_R = []
        self.alpha_list = []
        self.beta_list = []
        self.alpha_scaled = []
        self.beta_scaled = [] 

        for each_step in range(timestep):        
            # Apply GAT layers
            x = dynamic[:, each_step, :]  # Shape: [num_nodes, in_dim]
            
            x = self.conv1(x, self.g.edge_index)
            x = F.elu(x)
            x = self.conv2(x, self.g.edge_index)
            x = F.elu(x)
            
            # Aggregate node features (e.g., mean)
            cur_h = torch.mean(x, dim=0, keepdim=True)  # Shape: [1, hidden_dim2]
            
            h = self.gru(cur_h, h)
            hc = torch.cat((h, cI[each_step].reshape(1,1), cR[each_step].reshape(1,1)), dim=1)
            
            pred_I = self.nn_res_I(hc)
            pred_R = self.nn_res_R(hc)
            new_I.append(pred_I)
            new_R.append(pred_R)

            pred_res = self.nn_res_sir(hc)
            alpha = pred_res[:, 0]
            beta =  pred_res[:, 1]
            
            self.alpha_list.append(alpha)
            self.beta_list.append(beta)
            alpha = torch.sigmoid(alpha)
            beta = torch.sigmoid(beta)
            self.alpha_scaled.append(alpha)
            self.beta_scaled.append(beta)
            
            cur_phy_I = []
            cur_phy_R = []
            for i in range(self.pred_window):
                last_I = I[each_step] if i == 0 else last_I + dI.detach()
                last_R = R[each_step] if i == 0 else last_R + dR.detach()

                last_S = N - last_I - last_R
                
                dI = alpha * last_I * (last_S / N) - beta * last_I
                dR = beta * last_I
                cur_phy_I.append(dI)
                cur_phy_R.append(dR)
            cur_phy_I = torch.stack(cur_phy_I).to(self.device).permute(1,0)
            cur_phy_R = torch.stack(cur_phy_R).to(self.device).permute(1,0)

            phy_I.append(cur_phy_I)
            phy_R.append(cur_phy_R)

        new_I = torch.stack(new_I).to(self.device).permute(1,0,2)
        new_R = torch.stack(new_R).to(self.device).permute(1,0,2)
        phy_I = torch.stack(phy_I).to(self.device).permute(1,0,2)
        phy_R = torch.stack(phy_R).to(self.device).permute(1,0,2)

        self.alpha_list = torch.stack(self.alpha_list).squeeze()
        self.beta_list = torch.stack(self.beta_list).squeeze()
        self.alpha_scaled = torch.stack(self.alpha_scaled).squeeze()
        self.beta_scaled = torch.stack(self.beta_scaled).squeeze()
        return new_I, new_R, phy_I, phy_R, h
