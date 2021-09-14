import torch 
from torch.distributed import rpc 

from .framework import Euler_Encoder, Euler_Recurrent
from .utils import _remote_method, _remote_method_async

class StaticEncoder(Euler_Encoder):
    '''
    Static implimentation of Euler_Encoder interface
    '''

    def decode_all(self, zs, unsqueeze=False):
        '''
        Given node embeddings, return edge likelihoods for 
        all subgraphs held by this model
        For static model, it's very simple. Just return the embeddings
        for ei[n] given zs[n]

        zs : torch.Tensor
            A T x d x N tensor of node embeddings generated by the models, 
            it is safe to assume z[n] are the embeddings for nodes in the 
            snapshot held by this model's TGraph at timestep n
        '''
        assert not zs.size(0) < self.module.data.T, \
            "%s was given fewer embeddings than it has time slices"\
            % rpc.get_worker_info().name

        assert not zs.size(0) > self.module.data.T, \
            "%s was given more embeddings than it has time slices"\
            % rpc.get_worker_info().name

        preds = []
        ys = []
        cnts = []
        for i in range(self.module.data.T):
            preds.append(
                self.decode(self.module.data.eis[i], zs[i])
            )    
            ys.append(self.module.data.ys[i])
            cnts.append(self.module.data.cnt[i])

        return preds, ys, cnts

    def score_edges(self, z, partition, nratio):
        '''
        Given a set of Z embeddings, returns likelihood scores for all known
        edges, and randomly sampled negative edges

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
        n = self.module.data.get_negative_edges(partition, nratio)

        p_scores = []
        n_scores = []

        for i in range(len(z)):
            p = self.module.data.ei_masked(partition, i)
            if p.size(1) == 0:
                continue

            p_scores.append(self.decode(p, z[i]))
            n_scores.append(self.decode(n[i], z[i]))

        p_scores = torch.cat(p_scores, dim=0)
        n_scores = torch.cat(n_scores, dim=0)

        return p_scores, n_scores

    def calc_loss(self, z, partition, nratio):
        '''
        Sum up all of the loss per time step, then average it. For some reason
        this works better than running score edges on everything at once. It's better
        to run BCE per time step rather than all at once

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
        tot_loss = torch.zeros(1)
        ns = self.module.data.get_negative_edges(partition, nratio)

        for i in range(len(z)):
            ps = self.module.data.ei_masked(partition, i)
            
            # Edge case. Prevents nan errors when not enough edges
            # only happens with very small timewindows 
            if ps.size(1) == 0:
                continue

            tot_loss += self.bce(
                self.decode(ps, z[i]),
                self.decode(ns[i], z[i])
            )

        return tot_loss.true_divide(len(z))


class StaticRecurrent(Euler_Recurrent):
    def score_all(self, zs, unsqueeze=False):
        '''
        Has the distributed models score and label all of their edges
        Sends workers embeddings such that zs[n] is used to reconstruct graph at 
        snapshot n

        zs : torch.Tensor 
            A T x d x N tensor of node embeddings generated by each graph snapshot
            Need to offset according to how far in the future embeddings are supposed
            to represent.
        '''
        futs = []
        start = 0
    
        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            futs.append(
                _remote_method_async(
                    StaticEncoder.decode_all,
                    self.gcns[i],
                    zs[start : end],
                    unsqueeze=unsqueeze
                )
            )
            start = end 

        obj = [f.wait() for f in futs]
        scores, ys, cnts = zip(*obj)
        
        # Compress into single list of snapshots
        scores = sum(scores, [])
        ys = sum(ys, [])
        cnts = sum(cnts, [])

        return scores, ys, cnts


    def loss_fn(self, zs, partition, nratio=1):
        '''
        Runs NLL on each worker machine given the generated embeds
        Sends workers embeddings such that zs[n] is used to reconstruct graph at 
        snapshot n

        zs : torch.Tensor 
            A T x d x N tensor of node embeddings generated by each graph snapshot
            Need to offset according to how far in the future embeddings are supposed
            to represent.
        partition : int
            enum representing train, validation, test sent to workers
        nratio : float
            The workers sample nratio * |E| negative edges for calculating loss
        '''
        futs = []
        start = 0
    
        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            futs.append(
                _remote_method_async(
                    StaticEncoder.calc_loss,
                    self.gcns[i],
                    zs[start : end],
                    partition, nratio
                )
            )
            start = end 

        tot_loss = torch.zeros(1)
        for f in futs:
            tot_loss += f.wait()

        return [tot_loss.true_divide(self.num_workers)]
        

    def score_edges(self, zs, partition, nratio=1):
        '''
        Gets edge scores from dist modules, and negative edges. 
        Sends workers embeddings such that zs[n] is used to reconstruct graph at 
        snapshot n

        zs : torch.Tensor 
            A T x d x N tensor of node embeddings generated by each graph snapshot
            Need to offset according to how far in the future embeddings are supposed
            to represent.
        partition : int
            enum representing train, validation, test sent to workers
        nratio : float
            The workers sample nratio * |E| negative edges for calculating loss
        '''
        futs = []
        start = 0
    
        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            futs.append(
                _remote_method_async(
                    StaticEncoder.score_edges,
                    self.gcns[i],
                    zs[start : end], 
                    partition, nratio
                )
            )
            start = end 

        pos, neg = zip(*[f.wait() for f in futs])
        return torch.cat(pos, dim=0), torch.cat(neg, dim=0)