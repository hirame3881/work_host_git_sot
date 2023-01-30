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
                 supple_type=0,
                 batch_size=1,
                 softmax=False):
        super().__init__()
        self.controller_size = controller_size
        ##self.rrnn_mlp_out_size=memory_unit_size 
        self.rrnn_mlp_size=rel_config["rrnn_mlp_size"] 
        self.infer_flag=infer_flag
        self.sort_flag=sort_flag
        self.supple_type =  supple_type
        self.batch_size =batch_size
        self.softmax=softmax
        self.clip=20
        self.device=device

        if self.infer_flag:
            ##read_data_size=controller_size + num_heads * memory_unit_size+self.rrnn_mlp_out_size #test:memory_unit_size
            read_data_size=controller_size + num_heads * memory_unit_size+self.rrnn_mlp_size
            self.controller = NTMController(
                input_size + num_heads * memory_unit_size, controller_size, output_size,
                read_data_size=read_data_size, 
                device=device)
                
        else:
            self.controller = NTMController(
                input_size + num_heads * memory_unit_size, controller_size, output_size,
                read_data_size=controller_size + num_heads * memory_unit_size,
                device=device)
        
        #0:そのまま or sort 1:書き込み頻度全head 2:閾値?%-回数-全head 3:閾値?%ラスト書き込みタイミング-01正規化-全head 4:L付加 sumhead 5:L付加2つ
        if supple_type==0:
            rrnn_input_size = memory_unit_size
        elif supple_type>=1 and supple_type<=3:
            rrnn_input_size = memory_unit_size + num_heads
            #print("rrnn_input_size:",rrnn_input_size)
        if self.infer_flag:
            ##self.testlayer=nn.Linear(memory_unit_size*(memory_units+1),self.rrnn_mlp_out_size)
            self.infer_mod=RelationalMemory(mem_slots=rel_config["mem_slots"],
                                          head_size=int(rel_config["mem_params"] / (rel_config["num_heads"] * rel_config["mem_slots"])),
                                        input_size=rrnn_input_size ,
                                        dnc_nr_cells=memory_units,
                                        num_heads=rel_config["num_heads"],
                                        num_blocks=rel_config["num_blocks"],
                                        forget_bias=rel_config["forget_bias"], 
                                        input_bias=rel_config["input_bias"],
                                        attention_mlp_layers=rel_config["attention_mlp_layers"]
                                        )
            self.initial_infer_memory=self.infer_mod.initial_state(self.batch_size, trainable=True).to(self.device)
            self.infer_memory=self.initial_infer_memory
                #〜こちらのメモリはwrite()やread()で操作するのではなく、上書きしていく。これでは微分できないのでは？元コードもこう？　
                #元コードではforward_setpの結果を上書きして行ってたので大丈夫そう。なおforward最初=クエリ単位でrepackageが行われ、initial_stateはインスタンス作成時（epochの前）に一度行われるのみ
                #最初にinitial_stateで作ったmemoryはepochでは上書きされない(rrnn l206).memory変数として保持されっぱなし。forwardは初期化された空メモリとしてこれを受け取る
                #と思ったけどin-placeで更新とかされてるかもしれんしなるべく再現しよう。最初にここで宣言して触らず,resetタイミングでinfer_memにinitial_infer_memを代入
            self.rrnn_mlp=nn.Sequential(
                nn.Linear(rel_config["mem_params"], self.rrnn_mlp_size),
                nn.ReLU(),
                nn.Linear(self.rrnn_mlp_size, self.rrnn_mlp_size),
                nn.ReLU(),
                nn.Linear(self.rrnn_mlp_size, self.rrnn_mlp_size),
                nn.ReLU(),
                nn.Linear(self.rrnn_mlp_size, self.rrnn_mlp_size),
                nn.ReLU()
            )
            
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
        
        if softmax:self.softmax_layer=nn.Softmax(dim=1)
        
        self.prev_head_weights = []
        self.prev_reads = []
        if self.sort_flag:self.write_freq=torch.zeros(self.batch_size,self.memory_units).to(self.device)
        if self.supple_type==1 or self.supple_type==2:
            self.wfreq_perhead=torch.zeros(batch_size,self.memory_units,self.num_heads).to(self.device)
        if self.supple_type==3:
            self.last_w=torch.zeros(batch_size,self.memory_units,self.num_heads).to(self.device)
            self.time_count=0
        self.to(self.device)
        self.reset(batch_size=batch_size)

    def reset(self, batch_size=1):
        self.memory.reset(batch_size)
        self.controller.reset(batch_size)
        self.prev_head_weights = []
        for i in range(len(self.heads)):
            prev_weight = torch.zeros([batch_size, self.memory.n]).to(self.device)
            self.prev_head_weights.append(prev_weight)
        self.prev_reads = []
        for i in range(self.num_heads):
            prev_read = torch.zeros([batch_size, self.memory.m]).to(self.device)
            # using random initialization for previous reads
            nn.init.kaiming_uniform_(prev_read)
            self.prev_reads.append(prev_read)

        if self.sort_flag:
            '''self.write_freqs=[]
            for i in range(self.num_heads):
                write_freq = torch.zeros([batch_size, self.memory_units]).to(self.device)
                self.write_freqs.append(write_freq) #メモリは１つなのでwrite_freqはヘッド数ぶん持たなくていい '''
            self.write_freq=torch.zeros(batch_size,self.memory_units).to(self.device) ##head数に対応してない
        if self.supple_type==1 or self.supple_type==2:
            self.wfreq_perhead=torch.zeros(batch_size,self.memory_units,self.num_heads).to(self.device)
        if self.supple_type==3:
            self.last_w=torch.zeros(batch_size,self.memory_units,self.num_heads).to(self.device)
            self.time_count=0
        if self.infer_flag:
            atest=-2
            self.infer_memory=self.initial_infer_memory
            self.infer_memory= self.infer_mod.repackage_hidden(self.infer_memory) 
            

    def forward(self, in_data):
        controller_h_state, controller_c_state = self.controller(
            in_data, self.prev_reads)
        controller_h_state= torch.clamp(controller_h_state, -self.clip, self.clip)
        read_data = []
        head_weights = []
        add_v=torch.zeros(self.batch_size,1,self.memory_unit_size).to(self.device)
        
        if self.supple_type>=1 and self.supple_type<=3:whead_idx=0
        if self.supple_type==3:self.time_count+=1
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
                if self.supple_type==1:
                    self.wfreq_perhead[:,:,whead_idx] +=head_weight#.view(self.batch_size,self.memory_units,1)
                    whead_idx+=1
                if self.supple_type==2:
                    self.wfreq_perhead[:,:,whead_idx] +=(head_weight>0.1)#.view(self.batch_size,self.memory_units,1)
                    whead_idx+=1
                if self.supple_type==3:
                    updt_pos=head_weight>0.1
                    self.last_w[:,:,whead_idx] =self.last_w[:,:,whead_idx] *(self.time_count-1)
                    self.last_w[:,:,whead_idx] =self.last_w[:,:,whead_idx] * (~(updt_pos))
                    self.last_w[:,:,whead_idx] +=updt_pos * self.time_count
                    self.last_w[:,:,whead_idx] = self.last_w[:,:,whead_idx] / self.time_count
                    whead_idx+=1
                ##ここでmodeがwであるならaを返してもらわないとinfer modに入れられない。aはちゃんとbatchがdim0にある
                ##aはshapeとしては(batch,1,width)が想定らしいので治す。dim1はtimestep
                add_v+=add_v_perhead.unsqueeze(1)
            head_weights.append(head_weight)
                    
        if self.infer_flag:
            if self.sort_flag:
                sorted_freq, fai_idx = torch.topk(self.write_freq, self.memory_units, dim=1, largest=False)
                fai_rep=fai_idx.unsqueeze(2).repeat(1,1,self.memory_unit_size)
                sorted_memry=torch.gather(self.memory.memory,1,fai_rep)
                infer_input_memory=sorted_memry
            else:
                infer_input_memory=self.memory.memory
                #print("infer_input_memory size bef:",infer_input_memory.size())
            #print("st",type(self.supple_type))
            if self.supple_type==1 or self.supple_type==2:
                infer_input_memory =torch.cat((infer_input_memory, self.wfreq_perhead),dim=2) 
                #print("supple_type:",self.supple_type)
            if self.supple_type==3:
                infer_input_memory =torch.cat((infer_input_memory, self.last_w),dim=2)
                #print("last_w size:",self.last_w.size())
                #print("infer_input_memory size:",infer_input_memory.size())
            ##mlp_out =self.testlayer(torch.cat([infer_input_memory.view(-1,self.memory_units*self.memory_unit_size) ,add_v] ,dim=1) ) #mlp_out: (batch, rrnn_mlp_out_size)
            rrnn_out,self.infer_memory =  self.infer_mod.forward_step(infer_input_memory ,self.infer_memory,dnc_write_v=add_v,treat_input_as_matrix=True)
            ## r_out,r_mem =  self.relation_mod.forward_step(rrnn_input_after_gat,r_mem,dnc_write_v=dnc_write_v,treat_input_as_matrix=True)
            mlp_out =self.rrnn_mlp(rrnn_out) ##prev_r_out??
            ##本当はmlp,finaloutputするのは最後のoutだけ?

            output = self.controller.output(read_data+[mlp_out]) 

        else:
            output = self.controller.output(read_data)
        if self.softmax:output=self.softmax_layer(output)
        self.prev_head_weights = head_weights
        self.prev_reads = read_data

        return output
