from kge.model import KgeRgnnEncoder, KgeBase
from kge import Config, Configurable, Dataset
from kge.job import Job

from kge.model.embedder.rgnn_utils import *

import inspect
from torch import Tensor
import torch.nn

from typing import Optional, Tuple, Callable
from typing import TYPE_CHECKING


class MessagePassing(torch.nn.Module):
    def __init__(
        self, 
        dataset: Dataset, 
        aggregation: str = "add") -> "MessagePassing":
        r""" Base class for creating message passing layers.
        """
        super(MessagePassing, self).__init__()

        self.message_args = inspect.getargspec(self.message)[0][1:]	
        self.num_entities = dataset.num_entities()

    def propagate(
        self, edge_index: Tensor, edge_type: Tensor, aggregation: str = None, **kwargs
    ) -> Tensor:
        r""" It selects the embeddings for the given edges and computes the
        messages. When attention is used, messages are returned per edge. Else the
        aggregation to each node happens already  
        """

        kwargs["edge_index"] = edge_index

        # retrieve the embeddings for head, tail and relations of the messages,
        # as well as the other message arguments
        message_args = []
        for arg in self.message_args:
            if arg == "h_i": 
                h_i=torch.index_select(kwargs["x"], 0, edge_index[0])
                message_args.append(h_i) 
            elif arg == "h_j": 
                h_j= torch.index_select(kwargs["x"], 0, edge_index[1]) 
                message_args.append(h_j)	
            elif arg == "h_r":				
                h_r = torch.index_select(kwargs["r"], 0, edge_type) 
                message_args.append(h_r) 
            elif arg == "message_weight":
                if self.message_weight:
                    head_name = "head_" + str(kwargs["head"] + 1)
                    message_weight = torch.index_select(
                        self.weights["w_message_weight_{}".format(head_name)] , 0, edge_type
                    ) 
                    message_args.append(message_weight) 
                else:
                    message_args.append(None)
            elif arg == "alpha_r":
                if self.learned_relation_weight:
                    alphas = self.alpha(edge_type)
                    message_args.append(alphas)
            else:
                message_args.append(kwargs[arg])	

        out = self.message(*message_args)
        # aggregate the edge-level values per central node
        if self.aggregation:
            out = scatter_(aggregation, out, edge_index[0], 
                dim_size=self.num_entities)
        
        return out

    def message(
        self, 
        h_i, #: Tensor, 
        h_j, #: Tensor, 
        h_r, #: Tensor, 
        propagation_type, #: str, 
        edge_norm=None,  #Tensor
        mode="",  #str
        head=1, #int
        message_weight= None, # Tensor
        alpha_r=None): # Tensor
        r""" Computes the messages over the given edges with the specified
        mode. Returns one message per edge.    
        """
        if propagation_type == "per_relation_basis":
            # calculate the basis decomposition weight a_br*v_b
            weight = self._calculate_basis_weights(mode)
        elif propagation_type == "per_relation_block":
            weight = self._calculate_block_weights(mode)
        else:
            head_name = "_head_" + str(head + 1)
            weight 	= self.weights["w_{}".format(mode + head_name)] 
        # compute the composition phi
        composed  = self.composition(h_i, h_j, h_r, message_weight) 
        message_per_edge = torch.mm(composed, weight)
        if self.learned_relation_weight and propagation_type!= "loop":
            message_per_edge *= alpha_r 
        if edge_norm is not None: 
            message_per_edge *= edge_norm.view(-1, 1)
     
        return message_per_edge

    def composition(
        self, h_i: Tensor, h_j: Tensor, h_r: Tensor, *args, **kwargs
    ) -> Tensor:
        r""" Placeholder for the composition in the message.
        """
        return h_j

    def edge_norm(self, edge_index: Tensor, num_ent: int) -> Tensor:
        r""" computes the edge norm  \frac{1}{\sqrt{D_i}\sqrt{D_j}},
        where D_i and D_j are the degrees of nodes i and j of the edge.
        """
        row, col = edge_index 
        if self.propagation_type in ['per_relation_basis', 'per_relation_block']:
            # compute degree over the whole graph
            row_all, _ = self.edge_index
            edge_weight = torch.ones_like(row_all).float()
            deg	= scatter_add(edge_weight, row_all, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
            deg_inv	= deg.pow(-0.5)							# D^{-0.5}
            deg_inv[deg_inv	== float("inf")] = 0
            norm = deg_inv[row] * edge_weight[row] * deg_inv[col]
        else:
            edge_weight = torch.ones_like(row).float()
            deg	= scatter_add(edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
            deg_inv	= deg.pow(-0.5)							# D^{-0.5}
            deg_inv[deg_inv	== float("inf")] = 0
            norm = deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}
        return norm

    def _calculate_basis_weights(self, mode: str) -> Tensor:
        r"""Calculates the basis weights. 
        """
        if mode == "loop":
            return self.loop_weight

        if isinstance(mode, str):
            mode = int(mode)
        weight = torch.einsum(
            "bij,b->ij", self.bases, self.comps[mode]
        )
        return weight

    def _calculate_block_weights(self, mode: str) -> Tensor:
        r"""copied from the pytorch-rgcn (https://arxiv.org/pdf/2107.10015.pdf).
        Reshapes the block weights to arrange in a block-diagonal matrix.
        """
        if mode == "loop":
            return self.weights["w_{}".format(mode)] 
        #blocks = self.weights["w_{}".format(mode)]
        blocks = self.weights["w_blocks"][int(mode)] 
        dim = blocks.dim()
        siz0 = blocks.shape[:-3]
        siz1 = blocks.shape[-2:]
        blocks2 = blocks.unsqueeze(-2)
        eye = self._attach_dim(
            torch.eye(self.num_blocks_or_bases, device=self.device).unsqueeze(-2), 
            dim-3, 1
            )

        return (blocks2 * eye).reshape(
            siz0 + torch.Size(torch.tensor(siz1) * self.num_blocks_or_bases)
        )

    def _attach_dim(self, v, n_dim_to_prepend=0, n_dim_to_append=0):
        # copied from Pytorch RGCN (https://arxiv.org/pdf/2107.10015.pdf)
        return v.reshape(
            torch.Size([1] * n_dim_to_prepend) + v.shape + torch.Size([1] * n_dim_to_append)
        )

class MessagePassingLayer(MessagePassing):
    def __init__(
        self, 
        config: Config, 
        dataset: Dataset, 
        edge_index: Tensor, 
        edge_type: Tensor, 
        in_dim: int,
        out_dim: int, 
        weight_init: Callable, 
        bias_: bool, 
        bias_init: Callable,
        edge_dropout: float,
        self_edge_dropout: float, 
        rel_transformation: str,
        propagation: str,
        composition: str,
        message_weight: bool,
        learned_relation_weight: bool,
        attention: bool,
        num_heads: int,
        use_edge_norm: bool, 
        emb_propagation_dropout: float, 
        weight_decomposition: str = None, 
        num_blocks_or_bases: int = None) -> "MessagePassingLayer":
        r""" Rgnn Message Passing Layer. Flexible implementation of Rgnns Layers
         with different propagation types (weights), weight decompositions, 
         composition functions, aggregation functions and attention. Adapted
         and strongly modified from CompGCN (https://arxiv.org/pdf/1911.03082.pdf).
        """
        super(MessagePassingLayer, self).__init__(dataset)

        self.device	= config.get("job.device") 
        self.config = config	
        # data stats and adjacency indices
        self.dataset = dataset
        self.num_entities = dataset.num_entities()
        self.num_relations = dataset.num_relations() * 2 
        self.edge_index = edge_index
        self.edge_type = edge_type

        # input and output dimensions of the layer
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_edge_norm = use_edge_norm 

        # set parameter configurations
        self.weight_init = weight_init 
        self.bias_ = bias_
        if self.bias_: 
            self.bias = self._get_bias_param(bias_init)
        self.bn	= torch.nn.BatchNorm1d(self.out_dim).to(self.device) 
        self.weight_decomposition = weight_decomposition
        if self.weight_decomposition in ["block", "basis"]:
            if propagation == "per_relation":
                propagation = propagation + "_" + self.weight_decomposition
                self.num_blocks_or_bases = num_blocks_or_bases
            else:
                raise RuntimeError("Weight Decomposition is only supported for per_relation propagation")
                
        if composition[-8:] == "weighted" and message_weight==False:
            message_weight = True  
        self.message_weight = message_weight 
        self.learned_relation_weight = learned_relation_weight
        if self.learned_relation_weight:
            self.alpha = torch.nn.Embedding(self.num_relations + 1, 1)     
        self.propagation_type = propagation

        self.rel_transformation = rel_transformation 
        if self.rel_transformation == "linear":
            self.w_rel = self._get_param((self.in_dim, self.out_dim))

        # add relation for the self_loop
        self.loop_rel = self._get_param((1, self.in_dim)).to(self.device)

        # relation basis decomposition. 
        if self.weight_decomposition == "relation_basis":
            if num_blocks_or_bases < 1:
                raise ValueError(
                    f"At least 1 basis necessary for relation basis decomposition but {num_blocks_or_bases} given.")
            self.num_rel_bases = num_blocks_or_bases
            # basis decomposition parameters
            self.basis_vectors = self._get_param((self.num_rel_bases, self.in_dim))
            self.relation_basis_weights = self._get_param((self.num_relations, self.num_rel_bases)) 
        
        # embedding dropout: after propagation
        self.prop_drop = torch.nn.Dropout(emb_propagation_dropout)
        # edge and self-loop dropout
        self.edge_dropout = edge_dropout
        self.self_edge_dropout = self_edge_dropout

        # set attention parameters
        self.num_heads = num_heads
        self.attention = attention
        if self.attention:
            # at the moment very specific to RAGAT, can be made more configurable
            self.use_edge_norm = False
            self.leakyrelu = torch.nn.LeakyReLU(0.2)
            self.edge2node = Edge2Node() 
            self.aggregation = None # attention is computed over all edges, aggregation happens later
        else:
            self.aggregation = "add"

        # set weights and indices per mode according to the propagation type 
        self._set_propagation_weights() 

        self.graph_sampling = config.get("negative_sampling.graph_sampling")
        if not self.graph_sampling in ["uniform", "edge_neighbourhood"]:
            # only do it once if no graph sampling is used
            self._set_propagation_indices() 
        # retrieve the chosen composition function 
        self._set_composition_function(composition) 

    def forward(self, x: Tensor, r: Tensor) -> Tuple[Tensor, Tensor]: 
        r"""Runs the Message Passing forward pass. 
        """ 

        if self.weight_decomposition == "relation_basis":
            # combine bases with relation-specific combination
            r = torch.mm(self.relation_basis_weights, self.basis_vectors)

        # adjust indices to sampling size if graph sampling is in place
        if self.graph_sampling in ["uniform", "edge_neighbourhood"]:
            self.edge_index = self.dataset._indexes['edge_index']
            self.edge_type = self.dataset._indexes['edge_type']
            self._set_propagation_indices() 

        # add self-loop to the relation embeddings
        r = torch.cat([r, self.loop_rel], dim=0)

        # propagate the messages once per mode
        messages_per_head = dict()
        num_modes = len(self.modes)
        for head in range(self.num_heads):
            for mode in self.modes:
                # access the mode-specific edges
                node_index = self.node_indices[mode]
                rel_index = self.rel_indices[mode]
                # set and calculate correct norm
                norm = None
                if self.use_edge_norm and mode != "loop":
                    norm = self.edge_norm(node_index, self.num_entities) 
                # calculate message vectors and aggregate
                mode_message = self.propagate(
                    aggregation=self.aggregation, 
                    edge_index=node_index, 
                    edge_type=rel_index, 
                    x=x, 
                    r=r,
                    propagation_type=self.propagation_type, 
                    edge_norm=norm, 
                    mode=mode,
                    head=head
                )

                if self.attention: # RAGAT version
                    try:
                        messages = torch.cat([messages, mode_message]) 
                    except NameError:
                        # for first mode
                        messages = mode_message
                else: # don"t concat but sum messages from all three (CompGCN-style)
                    if mode == "loop":
                        if self.propagation_type == "direction":
                            mode_message_weighted = mode_message * (1/num_modes)
                        else:
                            mode_message_weighted = mode_message
                    else:
                        if self.propagation_type == "direction":
                            mode_message_weighted = self.prop_drop(mode_message) * (1/num_modes)
                        else:
                            mode_message_weighted = self.prop_drop(mode_message)
                    try:
                        messages += mode_message_weighted 
                    except NameError:
                        # for first mode
                        messages = mode_message_weighted
            messages_per_head[head] = messages
            del messages 

        if self.attention:
            # compute attended messages and average over heads
            for head in range(self.num_heads):
                attended_message = self._calculate_attended_message(messages_per_head[head], head) 
                attended_message = (1/self.num_heads) * attended_message  
                try:
                    attention_out += attended_message
                except NameError:
                    attention_out = attended_message  
            out = attention_out  
        else:
            out = messages_per_head[0]
                	
        if self.bias_:
            out = torch.add(out, self.bias)
        if self.propagation_type not in ['per_relation_basis', 'per_relation_block']:
            out = self.bn(out)

        # transform the relation (None or linear)
        rel = self._transform_relation(r)

        return out, rel	
		
    def _get_param(self, shape: Tuple) -> Tensor:
        r"""Create and initialise weight parameter of `shape`with `self.weight_init`.
        """ 
        param = torch.nn.Parameter(torch.empty(*shape, device=self.device))
        return self.weight_init(param)

    def _get_bias_param(self, bias_init: Callable) -> Tensor:
        r"""Register and initiliase bias parameter.
        """ 
        self.register_parameter(
            "bias", torch.nn.Parameter(torch.zeros(self.out_dim).to(self.device))
        ) 
        return bias_init(self.bias) 

    def _transform_relation(self, r: Tensor) -> Tensor:
        r"""transforms the relation embedding if rel_transformaton="linear.
        Returns r without self-loop relation."""
        
        if self.rel_transformation == "self":
            return r[:-1]
        elif self.rel_transformation == "linear":
            return torch.matmul(r, self.w_rel)[:-1]	
        else:
            raise NotImplementedError(f"""The relation transformation 
            {self.rel_transformation} is not supported. 
            Try self or linear.""")

    def _set_propagation_weights(self):
        r""" Sets the weights according to the propagation type. 
        For example, if the propagation is "direction", one weight matrix
        is created for the original edges, one for reciprocal edges and one for
        self-loops.
        "per_relation" creates one weight for each relation. Because this is
        computationally expensive, basis or block decomposition is
        applied.
        "single" has one weight for all relations, inclusively the
        self-loop, whereas "single_with_self_edge_weight" has two weights:
        one for all relations except self-loops and one for self-loops.  
        """
        
        self.weights = torch.nn.ParameterDict() 

        if self.propagation_type == "single":  
            modes = [""]
            self.self_edge_weight = False
        elif self.propagation_type == "single_with_self_edge_weight":
            modes = [""]
            self.self_edge_weight = True
        elif self.propagation_type == "direction":
            modes = ["in", "out"]
            self.self_edge_weight = True
        elif self.propagation_type in ["per_relation_basis", "per_relation_block"]:
            modes = [str(rel.item()) for rel in self.edge_type.unique()]
            self.self_edge_weight = True
            if self.propagation_type == "per_relation_basis":
                self.bases = self._get_param(
                    (self.num_blocks_or_bases, self.in_dim, self.out_dim)
                )
                self.comps = self._get_param(
                    (self.num_relations, self.num_blocks_or_bases)
                )
                self.loop_weight = self._get_param((self.in_dim, self.out_dim))
            else: # self.propagation_type == "per_relation_block"
                block_row_dim, row_modulo = divmod(self.in_dim, self.num_blocks_or_bases)
                block_col_dim, col_modulo = divmod(self.out_dim, self.num_blocks_or_bases)
                # check that the weight dims are divisible by the blocks
                if row_modulo != 0 or col_modulo != 0:
                    raise RuntimeError("""The dimension of the weight matrix
                    is not dividable by the number of blocks""")
                blocks = torch.nn.Parameter(
                    torch.empty((self.num_relations, self.num_blocks_or_bases, 
                    block_row_dim, block_col_dim), device=self.device))
                self.weights["w_blocks"] = schlichtkrull_normal_(
                    blocks, shape=[self.num_relations//2, block_row_dim])
                loop_block = torch.nn.Parameter(
                    torch.empty((self.in_dim, self.out_dim), device=self.device))
                self.weights["w_loop"] = schlichtkrull_normal_(
                    loop_block, shape=[self.num_relations//2, block_row_dim])        

        else:
            raise NotImplementedError(
                f"Propagation type {self.propagation_type} is not supported.")

        # add a separate weight matrix for the self-edge
        if self.self_edge_weight:
            modes += ["loop"] 

        # set weights for each attention head
        if self.propagation_type not in ["per_relation_basis", "per_relation_block"]: 
            for head in range(self.num_heads):
                head_name = "_head_" + str(head + 1)
                # set weights for each propagation mode
                for mode in modes:
                    # basis and block weights cannot fully be defined here as the weight dict 
                    # only takes parameters,
                    # not results of computations with parameters
                            self.weights["w_{}".format(mode + head_name)] = self._get_param(
                                (self.in_dim, self.out_dim))
                if self.message_weight:
                    self.weights["w_{}".format("message_weight" + head_name)] = self._get_param(
                        (self.num_relations + 1, self.in_dim)
                    )    
                if self.attention:
                    # specify weight for attention score computation
                    attention_head_name = "att_" + str(head + 1)
                    self.weights["w_{}".format(attention_head_name)] = self._get_param((self.out_dim, 1))

        self.modes = modes

    def _set_propagation_indices(self): 
        r""" Splits the edges according to the chosen modes and collects them
        in a dictionary. For example, if propagation happens `per_relation`,
        each relation gets an entry with all edges having the relation.
        """ 
        ## the right indices for the propagation of the weight types
        self.node_indices = dict()
        self.rel_indices = dict()

        # construct index for self-loop
        # creates an edge index to itself for every node, e.g. 3-3 or 156-156
        self_node_index = torch.stack(
            [torch.arange(self.num_entities), torch.arange(self.num_entities)]).to(self.device)
        self_rel_index = torch.full(
            (self_node_index.size(1),), self.num_relations, dtype=torch.long).to(self.device)
            
        # apply edge dropout
        num_edges_wo_loops = self.edge_index.size(1)
        edge_mask_one_direction = torch.rand(num_edges_wo_loops//2) > self.edge_dropout
        # This way corresponding reciprocal edges get dropped as well
        edge_mask = torch.cat([edge_mask_one_direction, edge_mask_one_direction])
        self.edge_index = self.edge_index[:, edge_mask]
        self.edge_type = self.edge_type[edge_mask]
   
        # apply self-edge dropout
        self_edge_mask = torch.rand(self.num_entities) > self.self_edge_dropout
        self_node_index_masked = self_node_index[:, self_edge_mask]
        self_rel_index_masked = self_rel_index[self_edge_mask]

        # construct full index with self-loop
        self.full_edge_index = torch.cat([self.edge_index, self_node_index_masked], dim=1)

        # slice edges according to modes of propagation type
        if self.propagation_type == "single":    
            self.node_indices[""] = torch.cat(
                [self.edge_index, self_node_index_masked], dim=1)            
            self.rel_indices[""] = torch.cat(
                [self.edge_type, self_rel_index_masked], dim=0)
            self.self_edge_weight = False

        elif self.propagation_type == "single_with_self_edge_weight": 
            self.node_indices[""] = self.edge_index
            self.rel_indices[""] = self.edge_type

        elif self.propagation_type == "direction":
            num_edges = self.edge_index.size(1) // 2

            self.node_indices["in"] = self.edge_index[:, :num_edges]
            self.node_indices["out"] = self.edge_index[:, num_edges:]

            self.rel_indices["in"] = self.edge_type[:num_edges]
            self.rel_indices["out"] = self.edge_type[num_edges:]

        elif self.propagation_type in ["per_relation_basis", "per_relation_block"]:
            for rel in self.edge_type.unique():
                # get the indices of the edges with the specific relation type
                rel = rel.item()
                rel_index = (self.edge_type == rel).nonzero(as_tuple=True)[0]
                self.node_indices[str(rel)] = self.edge_index[:, rel_index]
                self.rel_indices[str(rel)] = self.edge_type[rel_index] 
            self.modes = [str(rel.item()) for rel in self.edge_type.unique()] + ["loop"]
        else:
            raise NotImplementedError(f"Propagation {self.propagation_type} is not supported.")

        if self.self_edge_weight:    
            self.node_indices["loop"] = self_node_index_masked
            self.rel_indices["loop"] = self_rel_index_masked
    
    def _set_composition_function(self, composition_function: str) -> None:
        r""" Sets the correct composition function from the string. If
        `message_weight`is set to True, the weighted version of the message is used."""
        if composition_function[-8:] != "weighted" and self.message_weight==True:
            composition_function = composition_function + "_weighted"
            self.config.log(f"""composition function changed to {composition_function}
                  because message weight is set to {self.message_weight}""")
        if composition_function == "neighbour":
            composition_function = "neighbor"
        try:
            self.composition = globals()[composition_function]
        except Exception as e:
             raise NotImplementedError(
                 f"Composition Function {composition_function} not found.")

    def _calculate_attended_message(self, messages: Tensor, head: int) -> Tensor:
        r"""Computes attention weights over all relation-neighbour pairs of each
        node and weighs the messages with them.
        """ 
        # collect attention weight
        attention_weight_name = "att_" + str(head + 1)
        attention_weight = self.weights["w_{}".format(attention_weight_name)] 
        
        # calculate scores b_{irj} 
        scores = -self.leakyrelu(messages.mm(attention_weight).squeeze())
        # calculate exp(b_{irj})
        edge_exp = torch.exp(scores).unsqueeze(1)
        # aggregate exp(b_{irj}) --> j: sum_j(exp(b_{irj}))
        entity_exp = self.edge2node(
            self.full_edge_index, edge_exp, self.num_entities, 
            self.num_entities, 1, dim=1)
        entity_exp[entity_exp == 0.0] = 1.0
        # dropout on scores on all edges
        edge_exp = self.prop_drop(edge_exp)
        # am_{irj} = exp(b_{irj}) * m_{irj}
        weighted_message = edge_exp * messages
        # aggregate am_{irj} --> j: sum_j(am_{irj})
        out = self.edge2node(
            self.full_edge_index, weighted_message, self.num_entities,
            self.num_entities, out_features=self.out_dim, dim=1)
        # divide aggregated weighted message by sum of other exp scores to finish softmax
        out = out.div(entity_exp)
        assert not torch.isnan(out).any()
        return out

class TorchRgcnLayer(torch.nn.Module):
    def __init__(
        self,
        config: Config, 
        dataset: Dataset,
        in_dim: int,
        out_dim: int,
        edge_dropout: float,
        weight_init: Callable,
        bias_: bool,
        bias_init: Callable,
        self_edge_dropout: float = None,
        weight_decomposition: str = None,
        num_blocks_or_bases: int = None, 
        vertical_stacking: bool = True) -> "TorchRgcnLayer":
        r""" Layer closely adapted from the sparse matrix implementation of the
        Link Prediction torch-rgcn: https://arxiv.org/pdf/2107.10015.pdf.
        Uses a sparse, stacked adjacency matrix to speed up computation.

        """
        super(TorchRgcnLayer, self).__init__()

        self.device	= config.get("job.device") 

        # compute necessary data statistics
        self.dataset = dataset
        self.num_entities = dataset.num_entities()
        self.orig_num_relations = dataset.num_relations()
        self.num_relations = dataset.num_relations() * 2 + 1# with inverse edges TODO

        # set the graph indices and layer dimensions
        self.graph_sampling = config.get("negative_sampling.graph_sampling")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dropout = edge_dropout

        # set parameter configurations and initialise parameters 
        self.weight_init = weight_init
        self.bias_ = bias_
        self.bias_init = bias_init
        self.weight_decomposition = weight_decomposition
        self.num_blocks_or_bases =num_blocks_or_bases 
        if self.bias_: 
            self.bias = self._get_and_init_bias_param(bias_init)
        self._set_weights()

        # specify self edge dropout and adjacency stacking
        self.self_edge_dropout = self_edge_dropout
        self.vertical_stacking = vertical_stacking
        

    def forward(self, x: Tensor, r: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Runs the forward pass of R-GCN: for each relation-specific
        adjacency A_r and weight W_r A_rXW_r is computed and summed over
        all relations.
        """ 

        # set weights depending on the decomposition
        if self.weight_decomposition not in ["basis", "block"]:
            weights = self.weights
        elif self.weight_decomposition == "basis":
            weights = torch.einsum("rb, bio -> rio", self.comps, self.bases)
        elif self.weight_decomposition == "block":
            pass # the block weights are set differently depending on the stacking of A
        
        # check for graph sampling
        #if self.graph_sampling in ["edge_neighbourhood", "uniform"]:
        edge_index=self.dataset._indexes["edge_index"].to(self.device)
        edge_type=self.dataset._indexes["edge_type"].to(self.device)

        # apply edge dropout on direct and corresponding inverse edges
        # happens here because of the potential graph sampling
        num_edges_wo_loops = edge_index.size(1)
        edge_mask_one_direction = torch.rand(num_edges_wo_loops//2) > self.edge_dropout
        # This way corresponding reciprocal edges get dropped as well
        edge_mask = torch.cat([edge_mask_one_direction, edge_mask_one_direction])
        edge_index_dropped = edge_index[:, edge_mask]
        edge_type_dropped = edge_type[edge_mask]

        # add self edge (with applied dropout) to edge_index and edge_type
        edge_index_plus, edge_type_plus = self._add_self_edge(
            self.self_edge_dropout, edge_index_dropped, edge_type_dropped
        )

        # compute stacked indices (either vertical oder horizontal)
        adj_indices, adj_size = self._stack_adj_matrices(edge_index_plus, edge_type_plus)
        num_edges = adj_indices.size(0)
        vals = torch.ones(num_edges, dtype=torch.float, device=self.device)

        # apply normalisation depending on stacking
        sums = self._sum_sparse(adj_indices, vals, adj_size)
        if not self.vertical_stacking:
            # Rearrange column-wise normalised value to reflect original order (because of transpose-trick)
            #n = self.num_orig_edges
            #i = self.num_self_edges
            sums = torch.cat(
                [sums[self.num_orig_edges : 2*self.num_orig_edges], 
                 sums[:self.num_orig_edges], sums[-self.num_self_edges:]], dim=0)
        vals = vals / sums

        # build sparse adjacency matrix
        adj = torch.sparse.FloatTensor(
            indices=adj_indices.t(), values=vals, size=adj_size).to(self.device)

        # compute convolution
        if self.vertical_stacking:
            if self.weight_decomposition == "block":
                weights = self._calculate_block_weights(self.blocks)
                weights = torch.cat([weights, self.block_self.unsqueeze(0)], dim=0)
            AX = torch.spmm(adj, x)
            AX = AX.view(self.num_relations, self.num_entities, self.in_dim)
            out = torch.einsum("rio, rni -> no", weights, AX)
        else: # horizontal stacking
            if self.weight_decomposition == "block":
                num_rels = self.num_relations - 1
                block_x = x.view(
                    self.num_entities, self.num_blocks_or_bases, self.block_row_dim)
                XW = torch.einsum("nbi, rbio -> rnbo", block_x, self.blocks).contiguous()
                XW = XW.view(num_rels, self.num_entities, self.out_dim)
                self_XW = torch.einsum("ni, io -> no", x, self.block_self)[None, :, :]
                XW = torch.cat([XW, self_XW], dim=0)
                out = torch.mm(adj, XW.view(self.num_relations * self.num_entities, self.out_dim))
            else:
                XW = torch.einsum("ni, rio -> rno", x, weights).contiguous()
                out = torch.mm(adj, XW.view(self.num_relations * self.num_entities, self.out_dim))
        if self.bias is not None:
            out = torch.add(out, self.bias)
        
                # Checkpoint 5
        # print('min', torch.min(out))
        # print('max', torch.max(out))
        # print('mean', torch.mean(out))
        # print('std', torch.std(out))
        # print('size', out.size())
                
        return out, r

    def _set_weights(self):
        r"""Sets and initialises the weights according to the chosen
        decomposition method.

        """ 
        if self.weight_decomposition not in  ["basis", "block"]:
            if self.weight_decomposition != "None":
                print(f"""Decomposition {self.weight_decomposition} not supported. 
                Message Passing is executed with unrestrained weights.""")
            self.weights = self._get_and_init_w_param(
                (self.num_relations, self.in_dim, self.out_dim)
            )
            
        elif self.weight_decomposition == "basis":
            if self.num_blocks_or_bases <= 0:
                raise ValueError("Number of Bases has to be larger than zero for Basis Decomposition")
            else:
                self.bases = self._get_and_init_w_param(
                    (self.num_blocks_or_bases, self.in_dim, self.out_dim)
                )
                self.comps = self._get_and_init_w_param(
                    (self.num_relations, self.num_blocks_or_bases)
                )
                
        elif self.weight_decomposition == "block":
            self.block_row_dim, row_modulo = divmod(self.in_dim, self.num_blocks_or_bases)
            self.block_col_dim, col_modulo = divmod(self.out_dim, self.num_blocks_or_bases)
                # check that the weight dims are divisible by the blocks
            if row_modulo != 0 or col_modulo != 0:
                raise RuntimeError(f"""The dimensions of the weight matrix
                    {self.in_dim}, {self.out_dim} are not dividable by the 
                    number of blocks {self.num_blocks_or_bases}.""")
            # initialise blocks with schichtkrull_normal_
            blocks = torch.nn.Parameter(
                torch.empty((self.num_relations-1, self.num_blocks_or_bases, 
                self.block_row_dim, self.block_col_dim), device=self.device)
            )
            self.blocks = schlichtkrull_normal_(
                blocks, shape=[self.orig_num_relations, self.block_row_dim]
            )
            # checkpoint 3: block weights
            # print("min", torch.min(self.blocks))
            # print("max", torch.max(self.blocks))
            # print("mean", torch.mean(self.blocks))
            # print("std", torch.std(self.blocks))
            # print("size", self.blocks.size())
            block_self = torch.nn.Parameter(
                torch.empty((self.in_dim, self.out_dim), device=self.device)
            )
            self.block_self = schlichtkrull_normal_(
                block_self, shape=[self.orig_num_relations, self.block_row_dim]
            )
            # checkpoint 3: block weights
            # print("min", torch.min(self.block_self))
            # print("max", torch.max(self.block_self))
            # print("mean", torch.mean(self.block_self))
            # print("std", torch.std(self.block_self))
            # print("size", self.block_self.size())

    def _calculate_block_weights(self, blocks: Tensor) -> Tensor:
        r"""Reshapes the block weights to arrange in a block-diagonal matrix.
        """ 
        dim = blocks.dim()
        siz0 = blocks.shape[:-3]
        siz1 = blocks.shape[-2:]
        blocks2 = blocks.unsqueeze(-2)
        eye = self._attach_dim(
            torch.eye(self.num_blocks_or_bases, device=self.device).unsqueeze(-2), dim-3, 1
        )

        return (blocks2 * eye).reshape(
            siz0 + torch.Size(torch.tensor(siz1) * self.num_blocks_or_bases)
        )

    def _attach_dim(
        self, v: Tensor, n_dim_to_prepend:int = 0, n_dim_to_append:int = 0
    ) -> Tensor:
        # from Pytorch RGCN
        return v.reshape(
            torch.Size([1] * n_dim_to_prepend) + v.shape + torch.Size([1] * n_dim_to_append)
        )

    def _add_self_edge(
        self, self_edge_dropout: float, edge_index: Tensor, edge_type: Tensor
    ) -> Tuple[Tensor, Tensor]:
        r"""Add self-edges to the edge index and use self-edge dropout if specified
        """ 
        self_node_index = torch.stack(
            [torch.arange(self.num_entities), torch.arange(self.num_entities)]).to(self.device)
        self_rel_index = torch.full(
            (self_node_index.size(1),), self.num_relations-1, dtype=torch.long).to(self.device)
            # self-edge dropout
        self_edge_mask = torch.rand(self.num_entities) > self_edge_dropout
        self_node_index_masked = self_node_index[:, self_edge_mask]
        self_rel_index_masked = self_rel_index[self_edge_mask]

        self.num_orig_edges = edge_index.size(1)//2
        self.num_self_edges = self_rel_index_masked.size(0)
            
        edge_index_plus = torch.cat([edge_index, self_node_index_masked], dim=1)            
        edge_type_plus = torch.cat([edge_type, self_rel_index_masked], dim=0)
        
        return edge_index_plus, edge_type_plus


    def _get_and_init_w_param(self, shape: Tuple) -> Tensor: 
        r"""Create and initialise weight parameter of `shape`.
        """ 
        param = torch.nn.Parameter(torch.empty(*shape, device=self.device))
        param = self.weight_init(param)
        return param

    def _get_and_init_bias_param(self, bias_init: Callable) -> Tensor:
        r"""Register and initiliase bias parameter.
        """ 
        self.register_parameter(
            "bias", torch.nn.Parameter(torch.zeros(self.out_dim).to(self.device))
        ) 
        return bias_init(self.bias) 

    def _stack_adj_matrices(self, edge_index: Tensor, edge_type: Tensor):
        r""" Computes a sparse adjacency matrix for the given graph (the
        adjacency matrices of all relations are stacked vertically or
        horizontally).
        """
        
        n, r = self.num_entities, self.num_relations
        size = (r * n, n) if self.vertical_stacking else (n, r * n)

        fr, to = edge_index[0, :], edge_index[1, :] 
        offset = edge_type * n
        if self.vertical_stacking:
            fr = offset + fr
        else:
            to = offset + to

        indices = torch.cat([fr[:, None], to[:, None]], dim=1).to(self.device)

        return indices, size

    def _sum_sparse(self, indices, values, size):
        """
        Sum the rows or columns of a sparse matrix, and redistribute the
        results back to the non-sparse row/column entries
        Arguments are interpreted as defining sparse matrix.

        Source: 
        https://github.com/pbloem/gated-rgcn/blob/1bde7f28af8028f468349b2d760c17d5c908b58b/kgmodels/util/util.py#L304
        """

        assert len(indices.size()) == len(values.size()) + 1

        k, r = indices.size()

        if not self.vertical_stacking:
            # Transpose the matrix for column-wise normalisation
            indices = torch.cat([indices[:, 1:2], indices[:, 0:1]], dim=1)
            size = size[1], size[0]

        ones = torch.ones((size[1], 1), device=self.device)
        #if self.device == "cuda":
        #    values = torch.cuda.sparse.FloatTensor(indices.t(), values, torch.Size(size))
        #else:
        values = torch.sparse.FloatTensor(
            indices.t(), values, torch.Size(size)
        ).to(self.device)
        sums = torch.spmm(values, ones)
        sums = sums[indices[:, 0], 0]

        return sums # .view(k)

class WeightedGCNLayer(torch.nn.Module):
    def __init__(
        self,
        config: Config, 
        dataset: Dataset,
        edge_index: Tensor,
        edge_type: Tensor,
        in_dim: int,
        out_dim: int,
        weight_init: Callable,
        bias_: bool,
        bias_init: Callable,
        self_edge_dropout: float) -> "WeightedGCNLayer":
        r""" Layer closely adapted from the WGCN implementation 
        of https://arxiv.org/abs/1811.04441. Reduces adjacency dimension RxNxN to NxN
        by assigning each relation a weight and combining into one (sparse) adjacency matrix.
        """
        super(WeightedGCNLayer, self).__init__()

        self.device	= config.get("job.device") 
        num_edges = edge_index.size(1)
        self.edge_index = edge_index#[:, :num_edges//2]
        self.edge_type =  edge_type#[:num_edges//2]
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_entities = dataset.num_entities()
        self.num_relations = dataset.num_relations() * 2 + 1
        self.self_edge_dropout = self_edge_dropout  
        self.bias_ = bias_  
        self.weight = self._get_and_init_w_param((in_dim, out_dim), weight_init) 
        self.alpha = torch.nn.Embedding(self.num_relations + 1, 1, padding_idx=0)
        if bias_:
            self.bias = self._get_and_init_bias_param(bias_init)
        self.bn	= torch.nn.BatchNorm1d(self.out_dim).to(self.device) 

    def forward(self, x: Tensor, r: Tensor) -> Tuple[Tensor, Tensor]:
        r"""First reduces A \in \mathbb{R}^{RxNxN} to size NxN by giving each
        entry in A_r the weight alpha_r. Then computes the classic GCN
        convolution AXW.
        """ 
        # add self-edge
        edge_index, edge_type = self._add_self_edge(self.self_edge_dropout)
        # get the a_r per edge type
        alpha_r = self.alpha(edge_type).t()[0]
        # construct the sparse 2D adjacency matrix
        adj = torch.sparse_coo_tensor(
            edge_index, alpha_r, torch.Size((self.num_entities, self.num_entities)),
            requires_grad=True)
        # transform directed adjacency to undirected adjacency
        adj = adj + adj.transpose(0, 1) 
        XW = torch.mm(x, self.weight)
        out = torch.sparse.mm(adj, XW)
        if self.bias_:
            out = torch.add(out, self.bias)     
        out = self.bn(out)
        return out, r

    def _get_and_init_w_param(self, shape: Tuple, weight_init: Callable) -> Tensor:
        r"""Create and initiliase weight parameter with `shape`.
        """ 
        param = torch.nn.Parameter(torch.empty(*shape, device=self.device))
        param = weight_init(param)
        return param

    def _get_and_init_bias_param(self, bias_init: Callable) -> Tensor:
        self.register_parameter(
            "bias", torch.nn.Parameter(torch.zeros(self.out_dim).to(self.device))
        )
        return bias_init(self.bias)

    def _add_self_edge(self, self_edge_dropout=None) -> Tuple[Tensor, Tensor]:
        r"""Add self-edges to the edge index and use self-edge dropout if specified
        """ 
        self_node_index = torch.stack(
            [torch.arange(self.num_entities), torch.arange(self.num_entities)]
            ).to(self.device)
        self_rel_index = torch.full(
            (self_node_index.size(1),), self.num_relations-1, dtype=torch.long
        ).to(self.device)
        # self-edge dropout
        self_edge_mask = torch.rand(self.num_entities) > self_edge_dropout
        self_node_index_masked = self_node_index[:, self_edge_mask]
        self_rel_index_masked = self_rel_index[self_edge_mask]

        self.num_orig_edges = self.edge_index.size(1)//2
        self.num_self_edges = self_rel_index_masked.size(0)
            
        edge_index_plus = torch.cat([self.edge_index, self_node_index_masked], dim=1)            
        edge_type_plus = torch.cat([self.edge_type, self_rel_index_masked], dim=0)
        
        return edge_index_plus, edge_type_plus



class Rgnn(KgeBase):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        dim: int,
        init_for_load_only=False,
    ) -> "Rgnn":
        r"""Rgnn Model Class. Sets all configurations for the specific R-GNN
        model and computes the convolution in its forward method.
        """ 
        super().__init__(config, dataset, configuration_key)

        device = self.config.get("job.device")

        # edge and type indices
        # to.(device) necessary in case dataset is stored on another device
        edge_index = self.dataset.index("edge_index").to(device) 
        edge_type = self.dataset.index("edge_type").to(device)

        # data stats
        num_entities = self.dataset.num_entities()
        num_edges = edge_index.size(1)

        # layer input size and number of layers
        dim = dim
        num_layers = self.get_option("num_layers")
        
        # activation function
        act_key = self.get_option("activation")
        try:
            self.activation = getattr(torch, act_key)
        except Exception as e:
            raise ValueError(
                f"invalid activation function: {act_key}") from e
        
        # parameter configurations
        bias = self.get_option("bias")
        weight_decomposition = self.get_option("weight_decomposition")
        num_blocks_or_bases = self.get_option("num_blocks_or_bases")
        weight_init = self._find_init(self.get_option("weight_init"))
        if bias:
            bias_init = self._find_init(self.get_option("bias_init"))
        else:
            bias_init = None
        
        # hidden dropout (used on entity embeddings after each layer)
        self.emb_entity_dropout = torch.nn.Dropout(self.get_option("emb_entity_dropout"))
        
        # transformation of relation embedding
        rel_transformation = self.get_option("rel_transformation")

        # layer-type-specific options
        self.layer_type = self.get_option("layer_type")
        specific_layer_key = self.layer_type + "_args."
        if self.layer_type == "message_passing":
            composition = self.get_option(specific_layer_key + "composition")
            propagation = self.get_option(specific_layer_key + "propagation")
            message_weight = self.get_option(specific_layer_key + "message_weight")
            learned_relation_weight = self.get_option(specific_layer_key + "learned_relation_weight")
            use_edge_norm = self.get_option(specific_layer_key + "edge_norm")
            emb_propagation_dropout = self.get_option(specific_layer_key + "emb_propagation_dropout")
            attention = self.get_option(specific_layer_key + "attention")
            num_heads = self.get_option(specific_layer_key + "num_heads")
        elif self.layer_type == "torch_rgcn":
            vertical_stacking = self.get_option(specific_layer_key + "vertical_stacking")

        # Edge and self-edge dropout
        edge_dropout = self.get_option("edge_dropout")
        self_edge_dropout = self.get_option("self_edge_dropout")
        
        # Create module list with all layers
        self.gnn_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            # set the right in and out dimensions of the layer
            self._set_layer_dimensions(i, dim)    
            # in case of relation basis decomposition only apply it in the
            # first layer
            if weight_decomposition == "relation_basis" and i != 0:
                weight_decomposition = None
            # append layers according to layer type
            if self.layer_type == "message_passing":
                self.gnn_layers.append(
                    MessagePassingLayer(
                        self.config, 
                        self.dataset, 
                        edge_index, 
                        edge_type, 
                        self.in_dim, 
                        self.out_dim,
                        weight_init, 
                        bias, 
                        bias_init,
                        edge_dropout,
                        self_edge_dropout, 
                        rel_transformation,
                        propagation,
                        composition,  
                        message_weight,
                        learned_relation_weight,
                        attention,
                        num_heads,
                        use_edge_norm, 
                        emb_propagation_dropout,
                        weight_decomposition, 
                        num_blocks_or_bases  
                    )
                )
            elif self.layer_type == "torch_rgcn":
                self.gnn_layers.append(
                    TorchRgcnLayer(
                        self.config, 
                        self.dataset,
                        self.in_dim,
                        self.out_dim,
                        edge_dropout,
                        weight_init,
                        bias,
                        bias_init,
                        self_edge_dropout,
                        weight_decomposition,
                        num_blocks_or_bases,
                        vertical_stacking
                    )
                )
            elif self.layer_type == "weighted_gcn":
                self.gnn_layers.append(
                    WeightedGCNLayer(
                        self.config, 
                        self.dataset, 
                        edge_index, 
                        edge_type, 
                        self.in_dim, 
                        self.out_dim, 
                        weight_init, 
                        bias, 
                        bias_init, 
                        self_edge_dropout)
                )
            else:
                raise NotImplementedError(
                    f"""{self.layer_type} not supported. Use message_passing,
                    torch_rgcn or weighted_gcn.""")

    def forward(self, x: Tensor, r:Tensor) -> Tuple[Tensor, Tensor]:
        r"""Iterates over all R-GNN layers to transform the entity embeddings x
        and (optionally) the relation embeddings r.
        """ 
        for i in range(len(self.gnn_layers)):
            if self.layer_type == "torch_rgcn":
                # rgcn uses activation before the layer
                x = self.activation(x)
                # checkpoint 2: node embeddings before gcn layer
                # print("min", torch.min(x))
                # print("max", torch.max(x))
                # print("mean", torch.mean(x))
                # print("std", torch.std(x))
                # print("size", x.size())
            x, r = self.gnn_layers[i](x, r)
            if self.layer_type in ["message_passing", "weighted_gcn"]:
                x = self.activation(x)
            x = self.emb_entity_dropout(x)
        return x, r
        
    def _find_init(self, init_key: str) -> Callable: 
        r"""Looks up and returns the intitialisation function for `init_key`.
            Looks in torch.nn.init and rgnn_utils.py. Returns ValueError if it cannot be
            found in either.
        """ 
        try:
            # try to find it first from torch.nn.init
            init = getattr(torch.nn.init, init_key)
            return init
        except AttributeError:
            # then check user defined init functions (can also be imported from rgnn_utils)
            init = globals()[init_key]
            return init
        except Exception as e:
            raise ValueError(
                f"Invalid initialisation: {init_key}") from e

    def _set_layer_dimensions(self, layer_num: int, dim: int) -> None:
        r"""Sets the right layer dimensions. Takes the input dimension for the
        first layer from the embedders. Sets out-dimensions to the ones specified in
        the config if given. Else, the out-dimension is set to the in-dimension of the
        layer. The in-dimension of the subsequent layer is always set to the
        out-dimension of the last.
        """
        # ---- In-Dimension ---- #
        if layer_num == 0:
            self.in_dim = dim # take it from the embedders 
        else:
            # take the output dim of the previous layer as input dim
            self.in_dim = self.out_dim
        # ---- Out-Dimension ---- #
        try:
        # try to read from config, if not there take same dim as in_dim
            out_dim_key = str(layer_num + 1) + "_out_dim"
            self.out_dim = self.get_option(out_dim_key)
            if self.out_dim < 0:
                self.out_dim = self.in_dim
        except KeyError:
            self.out_dim = self.in_dim


class RgnnEncoder(KgeRgnnEncoder):
    def __init__(
        self, 
        config: Config, 
        dataset: Dataset, 
        configuration_key: str, 
        entity_embedder: "KgeEmbedder", 
        relation_embedder: "KgeEmbedder", 
        reciprocal_scorer=False, 
        init_for_load_only=False
    ) -> "RgnnEncoder":
        r"""Computes Relational Graph Neural Network Embeddings on top of the entity
        and relation embedders.
        """
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only
        )
        self.device = self.config.get("job.device")
        self.dataset = dataset

        # set the embedders for the original node and relation embeddings
        self.entity_embedder = entity_embedder
        self.relation_embedder = relation_embedder
        self.reciprocal_scorer = reciprocal_scorer

        # get the dimensions of the entity embedding to use for the first layer
        dim = self.entity_embedder.dim

        # create the Rgnn Model
        self.rgnn = Rgnn(
            self.config, self.dataset, self.configuration_key, dim
        )
        # create flag to avoid re-computation in case of negative sampling
        self.use_stale_embeddings = self.get_option("use_stale_embeddings")
        if self.use_stale_embeddings:
            self.computed = False

    def encode_spo(
        self, s: Tensor, p: Tensor, o: Tensor, direction=None, entity_subset=None
    ):
        r"""Computes embeddings over the graph and selects the embeddings for
        the batch.

        `s`, `p`, and `o` hold the indices of the current batch. 
        If the use of stale embeddings is specified in the Config, the
        `direction` information determines whether the R-GNN has to be computed.

        When the method is called in the score_sp_po() method of the RgnnModel,
        additionally selects the `entity_subset` and returns it.

        """
        if self.use_stale_embeddings:
            if direction in ["s", "sp_"]:
                if  self.computed == False:
                    self.ent_embeddings, self.rel_embeddings = self._run_rgnn()
                    self.computed = True
            elif direction in ["o", "p", "_po"]:
                self.computed = False
        else:
            self.ent_embeddings, self.rel_embeddings = self._run_rgnn()
        
        # select the embeddings for the current batch
        s, p, o = self._select_embeddings(s, p, o)

        # special selection for the .score_sp_po() method of the Rgnn
        if isinstance(entity_subset, torch.Tensor):
            ent_sub = torch.index_select(self.ent_embeddings, 0, entity_subset.long())
            return s, p, o, ent_sub
        elif entity_subset == "all":
            ent_sub = self.ent_embeddings
            return s, p, o, ent_sub
        # for all other scoring modes return batch embeddings of s, p and o
        else:
            return s, p, o

    def _run_rgnn(self) -> Tuple[Tensor, Tensor]:
        r"""Runs the forward pass of the R-GNN. In case the decoder is a
        reciprocal scorer, it keeps the embeddings of the inverse relations.

        """
        # compute convolution
        # checkpoint 1: node embeddings
        # print("min", torch.min(self.entity_embedder.embed_all()))
        # print("max", torch.max(self.entity_embedder.embed_all()))
        # print("mean", torch.mean(self.entity_embedder.embed_all()))
        # print("std", torch.std(self.entity_embedder.embed_all()))
        # print("size", self.entity_embedder.embed_all().size())

        ent_emb, rel_emb = self.rgnn.forward(
            self.entity_embedder.embed_all(), 
            self.relation_embedder.embed_all()
        )
        if not self.reciprocal_scorer:
            rel_emb = rel_emb[:self.dataset.num_relations()]

        return ent_emb, rel_emb

    def _select_embeddings(
            self, 
            s: Optional[Tensor]=None, 
            p: Optional[Tensor]=None, 
            o: Optional[Tensor]=None,
        ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Selects the embeddings of the current batch from all embeddings.
        If no index is specified for the position, returns all embeddings.

        """ 
        if s is not None:
            s = torch.index_select(self.ent_embeddings, 0, s.long())
        else:
            s = self.ent_embeddings
        if p is not None:
            p = torch.index_select(self.rel_embeddings, 0, p.long())
        else:
            p = self.rel_embeddings
        if o is not None:
            o = torch.index_select(self.ent_embeddings, 0, o.long())
        else:
            o = self.ent_embeddings
        
        return s, p, o
