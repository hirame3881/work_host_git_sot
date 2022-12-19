import torch
from torch import nn
import torch.nn.functional as F


class NTMController(nn.Module):

    def __init__(self, input_size, controller_size, output_size, read_data_size,device):
        super().__init__()
        self.input_size = input_size
        self.controller_size = controller_size
        self.output_size = output_size
        self.read_data_size = read_data_size
        self.device=device

        self.controller_net = nn.LSTMCell(input_size, controller_size)
        self.out_net = nn.Linear(read_data_size, output_size)
        # nn.init.xavier_uniform_(self.out_net.weight)
        nn.init.kaiming_uniform_(self.out_net.weight)
        self.h_state = torch.zeros([1, controller_size]).to(self.device)
        self.c_state = torch.zeros([1, controller_size]).to(self.device)
        # layers to learn bias values for controller state reset
        self.h_bias_fc = nn.Linear(1, controller_size)
        # nn.init.kaiming_uniform_(self.h_bias_fc.weight)
        self.c_bias_fc = nn.Linear(1, controller_size)
        # nn.init.kaiming_uniform_(self.c_bias_fc.weight)
        ##self.reset()

    def forward(self, in_data, prev_reads):
        #print("in-data:",in_data.device,"prev_reads[0]:",prev_reads[0].device)
        x = torch.cat([in_data] + prev_reads, dim=-1)
        self.h_state, self.c_state = self.controller_net(
            x, (self.h_state, self.c_state))
        return self.h_state, self.c_state

    def output(self, read_data):
        complete_state = torch.cat([self.h_state] + read_data, dim=-1)
        output = F.sigmoid(self.out_net(complete_state))
        return output

    def reset(self, batch_size=1):
        in_data = torch.tensor([[0.]]).to(self.device)  # dummy input
        ##print("h_bias_fc:",next(self.h_bias_fc.parameters()).device)
        h_bias = self.h_bias_fc(in_data)
        self.h_state = h_bias.repeat(batch_size, 1).to(self.device)
        c_bias = self.c_bias_fc(in_data)
        self.c_state = c_bias.repeat(batch_size, 1).to(self.device)
