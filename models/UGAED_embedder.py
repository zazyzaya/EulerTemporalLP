from copy import deepcopy

import torch 
from torch import nn 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.nn import MessagePassing

from .embedders import GCN 
from .framework import Euler_Embed_Unit
from .recurrent import EmptyModel
from .static import StaticEncoder, StaticRecurrent
from .utils import _remote_method, _remote_method_async, _param_rrefs

'''
Attempting strat from this paper:
Unified Graph Embedding-based Anomalous Edge Detection
https://ieeexplore.ieee.org/document/9206720
'''

class MeanGCN(GCN):
    def __init__(self, data_load, data_kws, h_dim, z_dim):
        super().__init__(data_load, data_kws, h_dim, z_dim)

        # Embeddings are averaged with neighbor embeddings 
        # to produce final output
        self.avg_neighbors = MessagePassing(aggr='mean')
        self.prob_distro = nn.Linear(h_dim, self.data.num_nodes)
        self.distros = []


    def inner_forward(self, mask_enum):
        '''
        Matrices are gonna be huge. Don't want to send them over
        the network. This will never be used for pred anyway so 
        we don't need to worry about alignment during decode
        '''
        self.distros = super().inner_forward(mask_enum)


    def forward_once(self, mask_enum, i):
        '''
        Essentially the same as the original fwd method, 
        but without final activation fn and with the 
        final message pass advocated by the authors of the 
        above paper
        '''
        # Get vars
        x = self.data.xs[i]
        ei = self.data.ei_masked(mask_enum, i)
        ew = self.data.ew_masked(mask_enum, i)

        # Simple 2-layer GCN. Tweak if desired
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)

        # Keep everything 2 layers
        #x = self.c2(x, ei, edge_weight=ew)

        # Add self loops manually 
        ei = torch.cat([
            ei, 
            torch.tensor(
                [list(range(self.data.num_nodes))]
            ).repeat(2,1)
        ], dim=1)
        x = self.avg_neighbors(ei, x=x, size=None)
        
        # Convert to probability distribution
        # where x[u,v] = P(v | u, N(u))
        x = torch.softmax(self.prob_distro(x))
        return x 

class UGAEDEncoder(StaticEncoder):
    def __init__(self, module: Euler_Embed_Unit, **kwargs):
        super().__init__(module, **kwargs)
        self.cross_entropy = nn.CrossEntropyLoss()

    def decode(self, e, z):
        '''
        Now expects z as a timecode and nothing more
        '''
        return self.distros[z][e[1], e[0]]
    
    def decode_all(self, zs):
        preds = []
        for i in range(len(self.module.distros)):
            e = self.module.data.eis[i]
            preds.append(self.decode(e, i))

    def calc_loss(self, z, partition, nratio):
        '''
        No negative sampling. Now it's a straight probability distro
        Observed samples scores' should drive down negative ones (I hope)
        '''

        # TODO if not to expensive, calc all at once to avoid averaging 
        losses = []
        for i in range(len(self.module.distros)):
            src,dst = self.module.data.ei_masked(partition, i)
            distros = self.module.distros[i][src]
            losses.append(
                self.cross_entropy(distros, dst)
            )

        return torch.stack(losses).mean()

    # Score_Edges should actually work with no changes since 
    # z is just a list of indices

    def forward(self, mask_enum, no_grad):
        # Saves all matrices here to avoid net traffic
        super().forward(mask_enum, no_grad)

        # Returns a list of indices so Leader gets to feel special
        return list(range(len(self.module.distros)))


class UGAED_Recurrent(StaticRecurrent):
    def __init__(self, rnn: nn.Module, remote_rrefs: list):
        # Just ignore whatever RNN user wants, we aren't using it
        super().__init__(EmptyModel, remote_rrefs)
        del self.rnn 

    def score_all(self, zs):
        '''
        Just ignore whatever zs we're given to process
        '''
        futs = []
        for i in range(self.num_workers):
            futs.append(
                _remote_method_async(
                    StaticEncoder.decode_all,
                    self.gcns[i],
                    None
                )
            )

        # From here out it's the same
        scores = [f.wait() for f in futs]
        ys = [
            _remote_method(
                StaticEncoder.get_data_field,
                self.gcns[i],
                'ys'
            ) for i in range(self.num_workers)
        ]

        return scores, ys

    # TODO 
    def loss_fn(self, zs, partition, nratio):
        pass 

    # TODO
    def score_edges(self, zs, partition, nratio):
        pass

    ###################################################################
    #    Stuff from the original framework that needed to be changed  #
    #    for a Non-rnn having model to work properly                  #
    ###################################################################

    def forward(self, mask_enum, include_h=False, h0=None, no_grad=False):
        '''
        Need matching signature, but won't use most of it
        '''
        futs = self.encode(mask_enum, no_grad)
        zs = [f.wait() for f in futs]

        # zs are just dummy lists, but it prob breaks if this 
        # is any different
        if include_h:
            return zs, None
        else:
            return zs

    def parameter_rrefs(self):
        '''
        Doesn't take into account RNN param anymore
        Just the workers parameters
        '''
        params = []
        for rref in self.gcns: 
            params.extend(
                _remote_method(
                    _param_rrefs, rref
                )
            )

        return params 

    def save_states(self):
        gcn = _remote_method(
            DDP.state_dict, self.gcns[0]
        )

        # Again, ignore any RNN params
        return gcn, None


    def load_states(self, gcn_state_dict, rnn_state_dict):
        # Ditto
        jobs = []
        for rref in self.gcns:
            jobs.append(
                _remote_method_async(
                    DDP.load_state_dict, rref, 
                    gcn_state_dict
                )
            )

        [j.wait() for j in jobs]

    