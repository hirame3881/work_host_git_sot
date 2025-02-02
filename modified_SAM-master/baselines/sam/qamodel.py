from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.sam.utils import MLP, LayerNorm, OptionalLayer
from baselines.sam.stm_basic import STM
from dnc.dnc import DNC


AVAILABLE_ELEMENTS = ('e1', 'e2', 'r1', 'r2', 'r3')




class QAmodel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        dnc_config=config["dnc"]
        rel_config =config["relationmod"]
        super(QAmodel, self).__init__()
        self.mlp_size = config["hidden_size"]
        self.rrnn_mlp_size=rel_config["rrnn_mlp_size"]
        self.out_plus_readVs_size=dnc_config["hidden_size"]+ dnc_config["cell_size"]*dnc_config["read_heads"]
        self.rrnn_type =dnc_config["rrnn_type"]
        
        self.input_module = InputModule(config)
        self.update_module = STM(config["symbol_size"], output_size=config["symbol_size"],
                                        init_alphas=[1, None, 0],
                                        learn_init_mem=True, mlp_hid=config['hidden_size'],
                                        num_slot=config["role_size"],
                                        slot_size=config["entity_size"],
                                        rel_size=96)

        self.infer_module = InferenceModule(config=config)
        ##self.lstm_module =nn.LSTM( config["symbol_size"],dnc_config["lstm_hidden"],num_layers =dnc_config["num_layers"],batch_first=True )
        self.dnc =DNC(
            input_size=config["symbol_size"],
            hidden_size=dnc_config["hidden_size"],
            num_hidden_layers=dnc_config["num_hidden_layers"],
            nr_cells= dnc_config["nr_cells"],
            cell_size= dnc_config["cell_size"],
            read_heads=dnc_config["read_heads"] ,
            gpu_id=0,
            rel_config=rel_config,
            gat_config=config["gat"],
            rrnn_type=dnc_config["rrnn_type"]
        )
        ##self.final_module=nn.Linear(dnc_config["lstm_hidden"],config["vocab_size"])
        ##self.final_module=nn.Linear(config["symbol_size"],config["vocab_size"])

        if dnc_config["rrnn_type"]:
            self.mlpmodule=nn.Sequential(
                nn.Linear(rel_config["mem_params"], self.rrnn_mlp_size),
                nn.ReLU(),
                nn.Linear(self.rrnn_mlp_size, self.rrnn_mlp_size),
                nn.ReLU(),
                nn.Linear(self.rrnn_mlp_size, self.rrnn_mlp_size),
                nn.ReLU(),
                nn.Linear(self.rrnn_mlp_size, self.rrnn_mlp_size),
                nn.ReLU()
            )
            self.final_module=nn.Linear(dnc_config["hidden_size"]+ dnc_config["cell_size"]*dnc_config["read_heads"]+self.rrnn_mlp_size,config["vocab_size"])
        else:
            self.final_module=nn.Linear(dnc_config["hidden_size"]+ dnc_config["cell_size"]*dnc_config["read_heads"],config["vocab_size"])
        self.Z = nn.Parameter(torch.zeros(config["entity_size"], config["vocab_size"]))
        nn.init.xavier_uniform_(self.Z.data)

    def forward(self, story: torch.Tensor, query: torch.Tensor):
        story_embed, query_embed = self.input_module(story, query)
        #out, (_,_,R) = self.update_module(story_embed.permute(1,0,2))
        #R = R.permute(0,2,1,3)
        #logits = self.infer_module(query_embed, R)
        query_view =query_embed.view(  query_embed.size()[0],1,-1 )
        storyquery=torch.cat((story_embed, query_view),1)
        ##_,lstm_out =self.lstm_module(storyquery)
        ##lstm_out =lstm_out[0]  #print(lstm_out[-1].size()) #(batch, lstm_hidden)
        dnc_out,_ = self.dnc(storyquery)  
        #print("dnc_out size:",dnc_out.size())
        dnc_out =dnc_out[:,-1] #(input + cell_size*read_heads) + rel_config["mem_params"]
        
        if self.rrnn_type:
            mlp_out =self.mlpmodule(dnc_out[:,self.out_plus_readVs_size:])
            dnc_rel_out =torch.cat([dnc_out[:,:self.out_plus_readVs_size] , mlp_out],1)
            logits=self.final_module(dnc_rel_out)
        else:
            logits=self.final_module(dnc_out)
        #print("logits size:",logits.size())
        return logits



class InputModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(InputModule, self).__init__()
        self.word_embed = nn.Embedding(num_embeddings=config["vocab_size"],
                                    embedding_dim=config["symbol_size"])
        nn.init.uniform_(self.word_embed.weight, -config["init_limit"], config["init_limit"])
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.ones(config["max_seq"], config["symbol_size"]))
        nn.init.ones_(self.pos_embed.data)
        self.pos_embed.data /= config["max_seq"]

    def forward(self, story: torch.Tensor, query: torch.Tensor):
        # Sentence embedding
        sentence_embed = self.word_embed(story)  # [b, s, w, e]
        sentence_sum = torch.einsum('bswe,we->bse', sentence_embed, self.pos_embed[:sentence_embed.shape[2]])
        # Query embedding
        query_embed = self.word_embed(query)  # [b, w, e]
        query_sum = torch.einsum('bwe,we->be', query_embed, self.pos_embed[:query_embed.shape[1]])
        return sentence_sum, query_sum



class InferenceModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(InferenceModule, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.ent_size = config["entity_size"]
        self.role_size = config["role_size"]
        self.symbol_size = config["symbol_size"]
        # output embeddings
        self.Z = nn.Parameter(torch.zeros(config["entity_size"], config["vocab_size"]))
        nn.init.xavier_uniform_(self.Z.data)

        # TODO: remove unused entity head?
        self.e = nn.ModuleList([MLP(equation='be,er->br', in_features=self.symbol_size,
                                    hidden_size=self.hidden_size, out_size=self.ent_size) for _ in range(2)])
        self.r = nn.ModuleList([MLP(equation='be,er->br', in_features=self.symbol_size,
                                    hidden_size=self.hidden_size, out_size=self.role_size) for _ in range(3)])
        self.l1, self.l2, self.l3 = [OptionalLayer(LayerNorm(hidden_size=self.ent_size), active=config["LN"])
                                     for _ in range(3)]

    def forward(self, query_embed: torch.Tensor, TPR: torch.Tensor):
        e1, e2 = [module(query_embed) for module in self.e]
        r1, r2, r3 = [module(query_embed) for module in self.r]

        i1 = self.l1(torch.einsum('be,br,berf->bf', e1, r1, TPR))
        i2 = self.l2(torch.einsum('be,br,berf->bf', i1, r2, TPR))
        i3 = self.l3(torch.einsum('be,br,berf->bf', i2, r3, TPR))

        step_sum = i1 + i2 + i3
        logits = torch.einsum('bf,fl->bl', step_sum, self.Z.data)
        return logits

