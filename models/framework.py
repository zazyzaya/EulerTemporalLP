from copy import deepcopy

import torch 
from torch import nn
from torch.distributed import rpc 
from torch.nn.parallel import DistributedDataParallel as DDP

from .utils import _remote_method, _remote_method_async, _param_rrefs

class Euler_Embed_Unit(nn.Module):
    '''
    Wrapper class to ensure calls to Embedders are formatted properly
    '''

    def inner_forward(self, mask_enum):
        '''
        The meat of the forward method. Runs data acquired from the worker
        through the mask_enum through whatever model it holds

        mask_enum : int
            enum representing train, validation, test sent to workers
        '''
        raise NotImplementedError

    def forward(self, mask_enum, no_grad):
        '''
        Forward method called by default. This ensures models can still use torch.no_grad() 
        with minimal extra hacking 

        mask_enum : int
            enum representing train, validation, test sent to workers
        no_grad : bool
            If true, tensor is returned without gradients for faster forward passes during eval
        '''
        if no_grad:
            with torch.no_grad():
                return self.inner_forward(mask_enum)
        
        return self.inner_forward(mask_enum)


class Euler_Encoder(DDP):
    '''
    Wrapper class for the DDP class that holds the data this module will operate on
    as well as a clone of the module itself

    Requirements: module must have a field called data containing all time slices it will
    operate on
    '''

    def __init__(self, module: Euler_Embed_Unit, **kwargs):
        '''
        Constructor for distributed encoder

        module : Euler_Embed_Unit
            The model to encode temporal data. module.forward must accept an enum 
            reprsenting train/val/test and nothing else. See embedders.py for acceptable
            modules 
        kwargs : dict
            any args for the DDP constructor
        '''
        super().__init__(module, **kwargs)

    
    def train(self, mode=True):
        '''
        This method is inacceessable in the DDP wrapped model by default
        '''
        self.module.train(mode=mode)

    
    def load_new_data(self, loader, kwargs):
        '''
        Put different data on worker. Must be called before work can be done
        
        loader : callable[..., loaders.TGraph]
            callable method that returns a loaders.TGraph object 
        kwargs : dict 
            kwargs for loader with a field for "jobs", the number of threads
            to load the TGraph with
        '''
        print(rpc.get_worker_info().name + ": Reloading %d - %d" % (kwargs['start'], kwargs['end']))
        
        jobs = kwargs.pop('jobs')
        self.module.data = loader(jobs, **kwargs)
        return True
    
    def get_data_field(self, field):
        '''
        Return some field from this worker's data object
        '''
        return self.module.data.__getattribute__(field)


    def decompress_scores(self, scores):
        '''
        Repeats scores as many times as that edge appears for better
        validation/testing. Otherwise edges that occur multiple times in
        a single timestep are compressed into a single weighted edge, and 
        multiple instances of the same sample are not evaluated accurately
        '''
        r_scores = []
        r_ys = []
        for i in range(self.module.data.T):
            counts = self.module.data.cnt[i]
            r_scores.append(torch.repeat_interleave(scores[i], counts, dim=0))
            r_ys.append(torch.repeat_interleave(self.module.data.ys[i], counts, dim=0))

        return r_scores, r_ys

    
    def run_arbitrary_fn(self, fn, *args, **kwargs):
        '''
        Run an arbitrary function using this machine
        '''
        return fn(*args, **kwargs)

    
    def decode(self, e,z):
        '''
        Given a single edge list and embeddings, return the dot product
        likelihood of each edge. Uses inner product by default

        e : torch.Tensor
            A 2xE list of edges where e[0,:] are the source nodes and e[1,:] are the dst nodes
        z : torch.Tensor
            A dxN list of node embeddings generated to represent nodes at this snapshot
        '''
        src,dst = e 
        return torch.sigmoid(
            (z[src] * z[dst]).sum(dim=1)
        )

    
    def bce(self, t_scores, f_scores):
        '''
        Computes binary cross entropy loss

        t_scores : torch.Tensor
            a 1-dimensional tensor of likelihood scores given to edges that exist
        f_scores : torch.Tensor
            a 1-dimensional tensor of likelihood scores given to edges that do not exist
        '''
        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        return (pos_loss + neg_loss) * 0.5


    def calc_loss(self, z, partition, nratio):
        '''
        Rather than sending edge index to master, calculate loss 
        on workers all at once. Must be implimented by the user

        z : torch.Tensor
            A T x d x N tensor of node embeddings generated by the models, 
            it is safe to assume z[n] are the embeddings for nodes in the 
            snapshot held by this model's TGraph at timestep n
        partition : int 
            An enum representing if this is training/validation/testing for 
            generating negative edges 
        nratio : float
            The model samples nratio * |E| negative edges for calculating loss
        '''
        raise NotImplementedError

    
    def decode_all(self, zs, unsqueeze=False):
        '''
        Given node embeddings, return edge likelihoods for all edges in snapshots held by this model. 
        Implimented differently for predictive and static models

        zs : torch.Tensor
            A T x d x N tensor of node embeddings generated by the models, 
            it is safe to assume z[n] are the embeddings for nodes in the 
            snapshot held by this model's TGraph at timestep n
        '''
        raise NotImplementedError
    
    
    def score_edges(self, z, partition, nratio):
        '''
        Scores all known edges and randomly sampled non-edges. The same as calc_loss but 
        does not return BCE, instead returns the actual scores given to edges

        z : torch.Tensor
            A T x d x N tensor of node embeddings generated by the models, 
            it is safe to assume z[n] are the embeddings for nodes in the 
            snapshot held by this model's TGraph at timestep n
        partition : int 
            An enum representing if this is training/validation/testing for 
            generating negative edges 
        nratio : float
            The model samples nratio * |E| negative edges for calculating loss
        '''
        raise NotImplementedError



class Euler_Recurrent(nn.Module):
    '''
    Abstract class for master module that holds all workers
    and calculates loss
    '''
    def __init__(self, rnn: nn.Module, remote_rrefs: list):
        '''
        Constructor for Recurrent layer of the Euler framework

        Parameters
        ------------
        rnn : torch.nn.Module
            An RNN-like module that accepts 3D tensors as input, and returns 
            T x d x N tensors of node embeddings for each snapshot
        remote_rrefs: list[torch.distributed.rpc.RRef]
            a list of RRefs to Euler_Workers

        Fields
        ------------
        gcns : list[torch.distributed.rpc.RRef]
            List of RRefs to workers
        rnn : torch.nn.Module
            The module used to process topological embeddings
        num_workers : int 
            The number of remote workers
        len_from_each : list[int]
            The number of snapshots held by each worker
        cutoff : float
            The threshold for anomalousness used by this model during classifiation
            Can be updated later, but defaults to 0.5
        '''
        super(Euler_Recurrent, self).__init__()

        self.gcns = remote_rrefs
        self.rnn = rnn 

        self.num_workers = len(self.gcns)
        self.len_from_each = []

        # Used for LR when classifying anomalies
        self.cutoff = 0.5


    def forward(self, mask_enum, include_h=False, h0=None, no_grad=False):
        '''
        First have each worker encode their data, then run the embeddings through the RNN 

        mask_enum : int
            enum representing train, validation, test sent to workers
        include_h : boolean
            if true, returns hidden state of RNN as well as embeddings
        h0 : torch.Tensor
            initial hidden state of RNN. Defaults to zero-vector if None
        no_grad : boolean
            if true, tells all workers to execute without calculating gradients.
            Used for speedy evaluation
        '''
        futs = self.encode(mask_enum, no_grad)

        # Run through RNN as embeddings come in 
        # Also prevents sequences that are super long from being encoded
        # all at once. (This is another reason to put extra tasks on the
        # workers with higher pids)
        zs = []
        for f in futs:
            z, h0 = self.rnn(
                f.wait(),
                h0, include_h=True
            )
            zs.append(z)
        
        #zs = [f.wait() for f in futs]

        # May as well do this every time, not super expensive
        self.len_from_each = [
            embed.size(0) for embed in zs
        ]
        zs = torch.cat(zs, dim=0)

        #zs, h0 = self.rnn(torch.cat(zs, dim=0), h0, include_h=True)

        if include_h:
            return zs, h0 
        else:
            return zs

    
    def encode(self, mask_enum, no_grad):
        '''
        Tell each remote worker to encode their data. Data lives on workers to minimize net traffic 

        mask_enum : int
            enum representing train, validation, test sent to workers
        no_grad : boolean
            if true, tells all workers to execute without calculating gradients.
            Used for speedy evaluation
        '''
        embed_futs = []
        
        for i in range(self.num_workers):    
            embed_futs.append(
                _remote_method_async(
                    DDP.forward, 
                    self.gcns[i],
                    mask_enum, no_grad
                )
            )

        return embed_futs


    def parameter_rrefs(self):
        '''
        Distributed optimizer needs RRefs to params rather than the literal
        locations of them that you'd get with self.parameters(). This returns
        a parameter list of all remote workers and an RRef of the RNN held by
        the recurrent layer
        '''
        params = []
        for rref in self.gcns: 
            params.extend(
                _remote_method(
                    _param_rrefs, rref
                )
            )
        
        params.extend(_param_rrefs(self.rnn))
        return params

   
    def save_states(self):
        '''
        Makes a copy of the current state dict as well as 
        the distributed GCN state dict (just worker 0)
        '''
        gcn = _remote_method(
            DDP.state_dict, self.gcns[0]
        )

        return gcn, deepcopy(self.state_dict())

    
    def load_states(self, gcn_state_dict, rnn_state_dict):
        '''
        Given the state dict for one GCN and the RNN load them
        into the dist and local models

        gcn_state_dict : dict  
            Parameter dict for remote worker 
        rnn_state_dict : dict
            Parameter dict for local RNN
        '''
        self.load_state_dict(rnn_state_dict)
        
        jobs = []
        for rref in self.gcns:
            jobs.append(
                _remote_method_async(
                    DDP.load_state_dict, rref, 
                    gcn_state_dict
                )
            )

        [j.wait() for j in jobs]

    
    def train(self, mode=True):
        '''
        Propogate training mode to all workers
        '''
        super(Euler_Recurrent, self).train() 
        [_remote_method(
            Euler_Encoder.train,
            self.gcns[i],
            mode=mode
        ) for i in range(self.num_workers)]


    def eval(self):
        '''
        Propogate training mode to all workers
        '''
        super(Euler_Recurrent, self).train(False)
        [_remote_method(
            Euler_Encoder.train,
            self.gcns[i],
            mode=False
        ) for i in range(self.num_workers)]

    
    def score_all(self, zs, unsqueeze=False):
        '''
        Has the distributed models score and label all of their edges
        Need to change which zs are given to workers depending on if 
        predictive or static

        zs : torch.Tensor 
            A T x d x N tensor of node embeddings generated by each graph snapshot
            Need to offset according to how far in the future embeddings are supposed
            to represent.
        '''
        raise NotImplementedError

    
    def loss_fn(self, zs, partition, nratio=1.0):
        '''
        Runs NLL on each worker machine given the generated embeds
        Need to change which zs are given to workers depending on if 
        predictive or static 

        zs : torch.Tensor 
            A T x d x N tensor of node embeddings generated by each graph snapshot
            Need to offset according to how far in the future embeddings are supposed
            to represent.
        partition : int
            enum representing train, validation, test sent to workers
        nratio : float
            The workers sample nratio * |E| negative edges for calculating loss
        '''
        raise NotImplementedError


    def score_edges(self, zs, partition, nratio=1):
        '''
        Gets edge scores from dist modules, and negative edges. Similar to 
        loss_fn but returns actual scores instead of BCE loss 

        zs : torch.Tensor 
            A T x d x N tensor of node embeddings generated by each graph snapshot
            Need to offset according to how far in the future embeddings are supposed
            to represent.
        partition : int
            enum representing train, validation, test sent to workers
        nratio : float
            The workers sample nratio * |E| negative edges for calculating loss
        '''
        raise NotImplementedError