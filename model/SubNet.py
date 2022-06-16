import torch
import torch.nn as nn
import torch.nn.functional as F
from model.lib import MLP, Encoder_MLP, Decoder_MLP, LayerNorm, Smooth_Encoder

class ActorSubNet(nn.Module):

    def __init__(self, args, hidden_size, depth=None):
        super(ActorSubNet, self).__init__()
        if depth is None:
            depth = 2

        # self.MLPs = nn.ModuleList([Encoder_MLP(8, 16), Encoder_MLP(16, 64), Encoder_MLP(64, 128)])
        self.Attn = nn.ModuleList([nn.MultiheadAttention(hidden_size, 8, dropout=0.1) for _ in range(depth)])
        self.Norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(depth)])
        self.smooth_traj = Smooth_Encoder(args, hidden_size)

    def forward(self, inputs, inputs_mask, polyline_mask, device):
        hidden_states_batch = inputs
        # hidden_states_batch = inputs[:, :, 4:]
        hidden_states_mask = inputs_mask

        # for i in range(3):
        #     hidden_states_batch = self.MLPs[i](hidden_states_batch)

        # 修改
        smooth_input = hidden_states_batch[polyline_mask]
        # smooth_input = hidden_states_batch
        smooth_output = self.smooth_traj(smooth_input)

        hidden_states_batch = torch.zeros((inputs.shape[0], inputs.shape[1], smooth_output.shape[2]), dtype=torch.float32).to(device)
        hidden_states_batch[polyline_mask] = smooth_output
        # hidden_states_batch = smooth_output
        
        for layer_index, layer in enumerate(self.Attn):
            temp = hidden_states_batch
            q = k = v = hidden_states_batch.permute(1,0,2)
            hidden_states_batch = layer(q, k, value=v, attn_mask=None, key_padding_mask=hidden_states_mask)[0].permute(1,0,2)  
            hidden_states_batch = hidden_states_batch + temp
            hidden_states_batch = self.Norms[layer_index](hidden_states_batch)
            hidden_states_batch = F.relu(hidden_states_batch)
            
        hidden_states_batch_mask = hidden_states_mask.unsqueeze(2).repeat(1, 1, hidden_states_batch.shape[2])
        hidden_states_batch_mask = hidden_states_batch_mask * -10000.0
        hidden_states_batch = hidden_states_batch + hidden_states_batch_mask
        hidden_states_batch = torch.max(hidden_states_batch, dim=1)[0]
        return hidden_states_batch

class MapSubNet(nn.Module):

    def __init__(self, args, hidden_size, depth=None):
        super(MapSubNet, self).__init__()
        if depth is None:
            depth = 2

        if args.data_type == 'lane_gcn':
            input_dim = 2
        else:
            input_dim = 8

        self.MLPs = nn.ModuleList([MLP(input_dim, hidden_size // 8), MLP(hidden_size // 4, hidden_size // 2)])
        self.Attn = nn.ModuleList([nn.MultiheadAttention(hidden_size // 8, 8, dropout=0.1), nn.MultiheadAttention(hidden_size // 2, 8, dropout=0.1)])
        self.Norms = nn.ModuleList([nn.LayerNorm(hidden_size // 4), nn.LayerNorm(hidden_size)])

        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs, inputs_mask, device):
        hidden_states_batch = inputs
        hidden_states_mask = inputs_mask

        for layer_index, layer in enumerate(self.Attn):
            hidden_states_batch = self.MLPs[layer_index](hidden_states_batch)
            temp = hidden_states_batch 
            q = k = v = hidden_states_batch.permute(1,0,2)
            hidden_states_batch = layer(q, k, value=v, attn_mask=None, key_padding_mask=hidden_states_mask)[0].permute(1,0,2)
            # hidden_states_batch = hidden_states_batch + temp
            hidden_states_batch = torch.cat([hidden_states_batch, temp], dim=2)
            hidden_states_batch = self.Norms[layer_index](hidden_states_batch)
            hidden_states_batch = F.relu(hidden_states_batch)
            
        hidden_states_batch_mask = hidden_states_mask.unsqueeze(2).repeat(1, 1, hidden_states_batch.shape[2])
        hidden_states_batch_mask = hidden_states_batch_mask * -10000.0
        hidden_states_batch = hidden_states_batch + hidden_states_batch_mask
        hidden_states_batch = torch.max(hidden_states_batch, dim=1)[0]
        return hidden_states_batch

# class SubNet(nn.Module):

#     def __init__(self, hidden_size, depth=None):
#         super(SubNet, self).__init__()
#         if depth is None:
#             depth = 3

#         self.MLPs = nn.ModuleList([Encoder_MLP(8, 16), Encoder_MLP(16, 64), Encoder_MLP(64, 128)])
#         self.Attn = nn.ModuleList([nn.MultiheadAttention(hidden_size, 8, dropout=0.1) for _ in range(depth)])
#         self.Norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(depth)])

#     def forward(self, inputs, inputs_mask, device):
#         hidden_states_batch = inputs
#         hidden_states_mask = inputs_mask

#         for i in range(3):
#             hidden_states_batch = self.MLPs[i](hidden_states_batch)
        
#         for layer_index, layer in enumerate(self.Attn):
#             temp = hidden_states_batch
#             q = k = v = hidden_states_batch.permute(1,0,2)
#             hidden_states_batch = layer(q, k, value=v, attn_mask=None, key_padding_mask=hidden_states_mask)[0].permute(1,0,2)
#             hidden_states_batch = F.relu(hidden_states_batch)
#             hidden_states_batch = hidden_states_batch + temp
#             hidden_states_batch = self.Norms[layer_index](hidden_states_batch)

#         hidden_states_batch_mask = hidden_states_mask.unsqueeze(2).repeat(1, 1, hidden_states_batch.shape[2])
#         hidden_states_batch_mask = hidden_states_batch_mask * -10000.0
#         hidden_states_batch = hidden_states_batch + hidden_states_batch_mask
#         hidden_states_batch = torch.max(hidden_states_batch, dim=1)[0]
#         return hidden_states_batch