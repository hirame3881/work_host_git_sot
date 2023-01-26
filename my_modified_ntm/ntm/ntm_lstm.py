import torch
from torch import nn

from .modules.controller import NTMController
from .modules.head import NTMHead
from .modules.memory import NTMMemory
from .relational_rnn_general import RelationalMemory

class NTM(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 controller_size,
                 memory_units,
                 memory_unit_size,
                 num_heads,
                 rel_config,
                 device,
                 infer_flag=False,
                 sort_flag=False,
                 batch_size=1,
                 softmax=False,
                 lstm_flag=False):
        super().__init__()
        self.controller_size = controller_size
        ##self.rrnn_mlp_out_size=memory_unit_size 
        self.rrnn_mlp_size=rel_config["rrnn_mlp_size"] 
        self.infer_flag=infer_flag
        self.sort_flag=sort_flag
        self.lstm_flag=lstm_flag
        self.batch_size =batch_size
        self.softmax=softmax
        self.clip=20
        self.device=device

        if self.lstm_flag:
            self.controller = NTMController(
                input_size,
                controller_size,
                output_size,
                read_data_size=controller_size,
                device=device)
        
        '''
        self.memory = NTMMemory(memory_units, memory_unit_size,self.device)
        self.memory_unit_size = memory_unit_size
        self.memory_units = memory_units
        self.num_heads = num_heads
        self.heads = nn.ModuleList([])
        for head in range(num_heads):
            self.heads += [
                NTMHead('r', controller_size, key_size=memory_unit_size),
                NTMHead('w', controller_size, key_size=memory_unit_size)
            ]
        '''
        
        if softmax:self.softmax_layer=nn.Softmax(dim=1)
        
        self.prev_head_weights = []
        self.prev_reads = []
        self.to(self.device)
        self.reset(batch_size=batch_size)

    def reset(self, batch_size=1):
        ##self.memory.reset(batch_size)
        self.controller.reset(batch_size)
        self.prev_head_weights = []
        '''for i in range(len(self.heads)):
            prev_weight = torch.zeros([batch_size, self.memory.n]).to(self.device)
            self.prev_head_weights.append(prev_weight)'''
        self.prev_reads = []
        '''for i in range(self.num_heads):
            prev_read = torch.zeros([batch_size, self.memory.m]).to(self.device)
            # using random initialization for previous reads
            nn.init.kaiming_uniform_(prev_read)
            self.prev_reads.append(prev_read)'''
            

    def forward(self, in_data):
        controller_h_state, controller_c_state = self.controller(
            in_data, self.prev_reads)
        controller_h_state= torch.clamp(controller_h_state, -self.clip, self.clip)
        read_data = []
        head_weights = []
        '''add_v=torch.zeros(self.batch_size,1,self.memory_unit_size).to(self.device)
        
        for head, prev_head_weight in zip(self.heads, self.prev_head_weights):
            if head.mode == 'r':
                ##print("sizes:",controller_c_state.size(), prev_head_weight.size(), self.memory.memory.size())
                head_weight, r ,_= head(
                    controller_c_state, prev_head_weight, self.memory)
                read_data.append(r)
            else:
                head_weight, _  ,add_v_perhead= head(
                    controller_c_state, prev_head_weight, self.memory)
                if self.sort_flag:self.write_freq+=head_weight
                ##ここでmodeがwであるならaを返してもらわないとinfer modに入れられない。aはちゃんとbatchがdim0にある
                ##aはshapeとしては(batch,1,width)が想定らしいので治す。dim1はtimestep
                add_v+=add_v_perhead.unsqueeze(1)
            head_weights.append(head_weight)'''
        output = self.controller.output(read_data)
        if self.softmax:output=self.softmax_layer(output)
        self.prev_head_weights = head_weights
        self.prev_reads = read_data

        return output
