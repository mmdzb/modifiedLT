import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from .SubNet import ActorSubNet, MapSubNet
from .lib import Encoder_MLP, Decoder_MLP, Goal_Decoder_MLP, Smooth_Encoder
from .TransformerDecoder import TransformerDecoder
from .LaneGCN_lib import PredNet

class modifiedLaneTransformer(nn.Module):
    def __init__(self, args):
        super(modifiedLaneTransformer, self).__init__()
        hidden_size = 128

        self.actor_net = ActorSubNet(args, hidden_size)
        self.map_net = MapSubNet(args, hidden_size)

        self.global_graph = nn.MultiheadAttention(hidden_size,8,dropout=0.1)   

        self.A2L = TransformerDecoder(hidden_size)
        self.L2A = TransformerDecoder(hidden_size)

        self.A2L_again = TransformerDecoder(hidden_size)
        self.L2A_again = TransformerDecoder(hidden_size)

        self.decoder = PredNet(args)

    def forward_encode_sub_graph(self,actor_total_input_padding, actor_input_mask, 
                    map_total_input_padding, map_input_mask,actor_polyline_mask,map_polyline_mask,device):
        ## 修改
        batch_size = actor_total_input_padding.shape[0]

        ## 修改
        actor_max_polyline_num = actor_total_input_padding.shape[1]
        map_max_polyline_num = map_total_input_padding.shape[1]

        hidden_size = 128

        if False:
            return actor_total_input_padding.mean() + map_total_input_padding.mean() + actor_input_mask.float().mean() + map_input_mask.float().mean() + actor_polyline_mask.float().mean() + map_polyline_mask.float().mean()

        # actor_input  = torch.flatten(actor_total_input_padding, start_dim=0, end_dim=1).to(device)
        actor_input = actor_total_input_padding.permute(2,3,0,1).flatten(2).permute(2,0,1)
        actor_input_mask = actor_input_mask.to(device)
        # map_input = torch.flatten(map_total_input_padding, start_dim=0, end_dim=1).to(device)
        map_input = map_total_input_padding.permute(2,3,0,1).flatten(2).permute(2,0,1)
        map_input_mask = map_input_mask.to(device)
        map_polyline_mask = map_polyline_mask.to(device)

        if False:
            return actor_input.mean() + map_input.mean() + actor_input_mask.float().mean() + map_input_mask.float().mean() + actor_polyline_mask.float().mean() + map_polyline_mask.float().mean()

        actor_states_batch = self.actor_net(actor_input, actor_input_mask, device)
        if False:
            return actor_states_batch + map_input.mean() + map_input_mask.float().mean() + actor_polyline_mask.float().mean() + map_polyline_mask.float().mean()
        map_states_batch = self.map_net(map_input, map_input_mask, device)
    
        actor_polyline_padding = actor_states_batch.view(batch_size, actor_max_polyline_num, hidden_size)
        actor_polyline_mask = actor_polyline_mask.to(device)
        map_polyline_padding  = map_states_batch.view(batch_size, map_max_polyline_num, hidden_size)
        
        #有改进空间
        ####################################
        actor_recover_padding = actor_polyline_mask.unsqueeze(2).repeat(1, 1, actor_polyline_padding.shape[2])
        actor_recover_padding = actor_recover_padding * 10000.0
        actor_polyline_padding = actor_polyline_padding + actor_recover_padding

        map_recover_padding = map_polyline_mask.unsqueeze(2).repeat(1, 1, map_polyline_padding.shape[2])
        map_recover_padding = map_recover_padding * 10000.0
        map_polyline_padding = map_polyline_padding + map_recover_padding
        ####################################

        lanes = map_polyline_padding.permute(1, 0, 2)
        lanes_mask = map_polyline_mask
        agents = actor_polyline_padding.permute(1, 0, 2)
        agents_mask = actor_polyline_mask
        lanes = lanes + self.A2L(lanes, lanes_mask, agents, agents_mask)
        agents = agents + self.L2A(agents, agents_mask, lanes, lanes_mask)

        lanes = lanes + self.A2L_again(lanes, lanes_mask, agents, agents_mask)
        agents = agents + self.L2A_again(agents, agents_mask, lanes, lanes_mask)
 
        return agents.permute(1, 0, 2)

    def forward(self, actor_total_input_padding, actor_input_mask, 
                    map_total_input_padding, map_input_mask,actor_polyline_mask,map_polyline_mask,global_graph_mask):
        ## 修改
        device = actor_total_input_padding.device
        bs = actor_total_input_padding.shape[0]
        if False:
            return actor_total_input_padding.mean() + map_total_input_padding.mean() + actor_input_mask.float().mean() + map_input_mask.float().mean() + actor_polyline_mask.float().mean() + map_polyline_mask.float().mean() + global_graph_mask.float().mean()
        agent_states_batch = self.forward_encode_sub_graph(actor_total_input_padding, actor_input_mask, 
            map_total_input_padding, map_input_mask,actor_polyline_mask,map_polyline_mask,device)
        if False:
            return agent_states_batch + global_graph_mask.float().mean()

        inputs = agent_states_batch.permute(1, 0, 2)
        inputs_mask = global_graph_mask.to(device)
        hidden_states = self.global_graph(query=inputs, 
                                    key=inputs,
                                    value=inputs,
                                    key_padding_mask=inputs_mask
                                    )[0]
        hidden_states = hidden_states.permute(1, 0, 2)

        if False:
            return hidden_states + global_graph_mask.float().mean()

        a, b = [], []
        for i in range(bs):
            a.append(torch.tensor(i).to(device))
            b.append(torch.tensor([0, 0],dtype=torch.float32).to(device))

        if False:
            return hidden_states + global_graph_mask.float().mean()

        out = self.decoder(hidden_states[:, 0, :], a, b)
        
        return torch.stack(out['reg'])

## python export_modified.py --eval --cfg configs/modifiedLaneTransformer.yaml --resume ./weights/modifiedLaneTransformer.pth --batch-size-onnx 32
## trtexec --onnx=./weights/modifiedLaneTransformer.onnx --buildOnly --verbose --saveEngine=./weights/modifiedLaneTransformer.engine --workspace=4096 --minShapes=actor_total_input_padding:64x1x19x8,actor_input_mask:64x19,map_total_input_padding:64x1x9x8,map_input_mask:64x9,actor_polyline_mask:64x1,map_polyline_mask:64x1,global_graph_mask:64x1 --optShapes=actor_total_input_padding:64x50x19x8,actor_input_mask:3200x19,map_total_input_padding:64x50x9x8,map_input_mask:3200x9,actor_polyline_mask:64x50,map_polyline_mask:64x50,global_graph_mask:64x50 --maxShapes=actor_total_input_padding:64x100x19x8,actor_input_mask:6400x19,map_total_input_padding:64x100x9x8,map_input_mask:6400x9,actor_polyline_mask:64x100,map_polyline_mask:64x100,global_graph_mask:64x100

