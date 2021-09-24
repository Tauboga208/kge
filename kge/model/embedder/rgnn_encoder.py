from kge.model import KgeRgnnEncoder, KgeBase
from kge import Config, Dataset
from kge.job import Job

import inspect
import torch.nn
# from torch.nn.init import xavier_normal_

from typing import Optional, Tuple



class MessagePassing(torch.nn.Module):
    """Base class for creating message passing layers. """

    def __init__(self, dataset, aggregation="add"):
        super(MessagePassing, self).__init__()

        self.message_args = inspect.getargspec(self.message)[0][1:]	# In the defined message function: get the list of arguments as list of string| For eg. in rgcn this will be ['x_j', 'edge_type', 'edge_norm'] (arguments of message function)
        self.update_args  = inspect.getargspec(self.update)[0][2:]
        self.num_entities = dataset.num_entities()

    def propagate(self, aggregation, edge_index, edge_type, **kwargs):

        kwargs['edge_index'] = edge_index

        # retrieve the embeddings for head, tail and relations of the messages,
        # as well as the other message arguments
        message_args = []
        for arg in self.message_args:
            if arg == 'h_i': 
                h_i=torch.index_select(kwargs['x'], 0, edge_index[0])
                message_args.append(h_i) 
            elif arg == 'h_j': 
                h_j= torch.index_select(kwargs['x'], 0, edge_index[1]) 
                message_args.append(h_j)	
            elif arg == 'h_r':				
                h_r = torch.index_select(kwargs['r'], 0, edge_type) 
                message_args.append(h_r) 
            else:
                message_args.append(kwargs[arg])	

        update_args = [kwargs[arg] for arg in self.update_args]	

        out = self.message(*message_args)
        # aggregate the edge-level values per entity
        out = scatter_(aggregation, out, edge_index[0], dim_size=self.num_entities)
        out = self.update(out, *update_args)
        
        return out

    def message(self, h_i, h_j, h_r, propagation_type, edge_norm=None, mode=""):

        # TODO attention
        if propagation_type == "per_relation_basis":
            # calculate the basis decomposition weight a_br*v_b
            weight = self._calculate_basis_weights(mode)
        elif propagation_type == "per_relation_block":
            weight = self._calculate_block_weights(mode)
        else:
            weight 	= self.weights['w_{}'.format(mode)] 
        # compute the composition phi
        composed  = self.composition(h_i, h_j, h_r) 
        message_per_edge = torch.mm(composed, weight)
        if edge_norm is not None: 
            message_per_edge *= edge_norm.view(-1, 1)
     
        return message_per_edge

    def update(self, aggr_out, *args, **kwargs):
        return aggr_out

    def composition(self, h_i, h_j, h_r, *args, **kwargs):
        return h_j

    def edge_norm(self, edge_index, num_ent):
        # computes the edge norm of the currently propagated index
        # TODO: r-gcn hat 1/c_i statt 1/c_i,r?
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg	= scatter_add(edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
        deg_inv	= deg.pow(-0.5)							# D^{-0.5}
        deg_inv[deg_inv	== float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}
        
        return norm

    def _calculate_basis_weights(self, mode):
        if mode == "loop":
            return self.loop_weight

        if isinstance(mode, str):
            mode = int(mode)
        weight = torch.einsum(
            'bij,b->ij', self.basis_matrices, 
            self.block_rel_basis_weights[mode]
        )
        return weight

    def _calculate_block_weights(self, mode):
        # copied from the pytorch-rgcn
        if mode == "loop":
            return self.weights['w_{}'.format(mode)] 

        blocks = self.weights['w_{}'.format(mode)] 
        dim = blocks.dim()

        siz0 = blocks.shape[:-3]
        siz1 = blocks.shape[-2:]

        blocks2 = blocks.unsqueeze(-2)

        eye = self._attach_dim(
            torch.eye(self.num_blocks_or_bases, device=self.device).unsqueeze(-2), dim - 3, 1
            )

        return (blocks2 * eye).reshape(
            siz0 + torch.Size(torch.tensor(siz1) * self.num_blocks_or_bases)
        )

    def _attach_dim(self, v, n_dim_to_prepend=0, n_dim_to_append=0):
        # from Pytorch RGCN
        return v.reshape(
            torch.Size([1] * n_dim_to_prepend) + v.shape + torch.Size([1] * n_dim_to_append)
        )

# 1 Build the layer
# in_channels and out_channels are the embedding dim
class RgnnLayer(MessagePassing):
    def __init__(
        self, config, dataset, use_edge_norm, 
        emb_propagation_dropout, emb_entity_dropout, bias, propagation_type, 
        in_dim, out_dim, composition_function, activation, rel_transformation, 
        edge_mask, self_edge_mask, weight_init, num_rel_bases=None, num_blocks_or_bases=None):
        super(self.__class__, self).__init__(dataset)

        self.device	= config.get('job.device') 	
        # data stats and adjacency indices
        self.num_entities = dataset.num_entities()
        self.num_relations = dataset.num_relations() * 2 # with inverse edges TODO
        self.edge_index = dataset.index("edge_index")
        self.edge_type = dataset.index("edge_type")
        # input and output dimensions of the layer
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.rel_transformation = rel_transformation  
        self.act = activation      
        self.weight_init = weight_init # TODO make more flexible
        self.bias_ = bias
        self.use_edge_norm = use_edge_norm

        if self.bias_: 
            self.register_parameter(
                "bias", torch.nn.Parameter(torch.zeros(self.out_dim).to(self.device))
            ) 
        else:
            self.bn	= torch.nn.BatchNorm1d(self.out_dim).to(self.device)
        
        self.propagation_type = propagation_type
        self.w_rel = self._get_param((self.in_dim, self.out_dim))
        self.loop_rel = self._get_param((1, self.in_dim)).to(self.device)

        # decomposition. TODO: only execute when decomposition is on 
        self.num_rel_bases = num_rel_bases
        # basis relation vector
        if self.num_rel_bases > 0:
            self.basis_vectors = self._get_param((self.num_rel_bases, self.in_dim))
            self.relation_basis_weights = self._get_param((self.num_relations, self.num_rel_bases))
        self.num_blocks_or_bases = num_blocks_or_bases

        self._set_propagation_weights() # set the weights per mode according to the propagation type 
        self._set_propagation_indices() # set the indices per mode according to the propagation type 
        self._set_composition_function(composition_function) # TODO: getattr and helper

        # embedding dropouts
        # after propagation
        self.prop_drop = torch.nn.Dropout(emb_propagation_dropout)
        # after the layer
        self.entity_drop = torch.nn.Dropout(emb_entity_dropout)

        # edge and self-loop dropout
        self.edge_mask = edge_mask
        self.self_edge_mask = self_edge_mask  
		
    def forward(self, x, r): 

        # add self-loop to the relation embeddings
        if self.num_rel_bases > 0:
            r = torch.mm(self.relation_basis_weights, self.basis_vectors)
        r = torch.cat([r, self.loop_rel], dim=0)
        # propagate the messages once per mode
        
        num_modes = len(self.modes)
        # num_edges = len(self.edge_type)
        out = torch.zeros((self.num_entities, self.out_dim)).to(self.device)
        for mode in self.modes:
            node_index = self.node_indices[mode]
            rel_index = self.rel_indices[mode]
            norm = None
            if self.use_edge_norm and mode != "loop":
                norm = self.edge_norm(node_index, self.num_entities) 
            mode_message = self.propagate(
                'add', node_index, rel_index, x=x, r=r, 
                propagation_type = self.propagation_type, 
                edge_norm=norm, mode=mode
            )
            if mode == "loop":
                # hier kommt dann wahrsch. self-edge-drop hin
                #mode_message_weighted = self.self_loop_drop(mode_message) *
                #(1/num_modes)
                mode_message_weighted = mode_message * (1/num_modes)
            else:
                mode_message_weighted = self.prop_drop(mode_message) * (1/num_modes)
            # TODO: look up why no dropout on self-messages or how to implement
            # it flexibly --> dropout mode dict probably 
            out += mode_message_weighted # TODO allow other aggregations
            # TODO think: is this already some kind of attention; node rel weighting?
		
        if self.bias_: 
            out = out + self.bias
        else: # TODO: möglich machen keins von beiden
            out = self.bn(out)
        out = self.act(out)
        out = self.entity_drop(out)

        rel = self._transform_relation(r)
        
        # TODO: make relation transformation more flexible
        # print("Check that rGNN is computed")
        return out, rel	# Ignoring the self loop inserted 
		
    def _get_param(self, shape):
        param = torch.nn.Parameter(torch.empty(*shape, device=self.device))
        self.weight_init(param)
        return param

    def _transform_relation(self, r):
        if self.rel_transformation == "self":
            return r[:-1]
        elif self.rel_transformation == "linear":
            return torch.matmul(r, self.w_rel)[:-1]	
        else:
            raise NotImplementedError("""The relation transformation is not
            supported. Try self, linear or convoluted.""")

    def _set_propagation_weights(self):
        
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
                self.basis_matrices = self._get_param(
                    (self.num_blocks_or_bases, self.in_dim, self.out_dim)
                )
                self.block_rel_basis_weights = self._get_param(
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

        else:
            raise NotImplementedError(
                "The specified propagation type is not supported: {}".format(self.propagation_type)
            )

        # add a separate weight matrix for the self-edge
        if self.self_edge_weight:
            modes += ["loop"] 
 
        for mode in modes:
            # cannot be defined here as the weight dict only takes parameters,
            # computed tensors
            if not self.propagation_type == "per_relation_basis": 
                if self.propagation_type == "per_relation_block":
                    if mode == "loop":
                        self.weights["w_{}".format(mode)] = self._get_param(
                            (self.in_dim, self.out_dim)
                        )
                    else:
                        # define the blocks for the block diagonal matrix
                        self.weights["w_{}".format(mode)] = self._get_param(
                            (self.num_blocks_or_bases, block_row_dim, block_col_dim)
                        )
                else:
                    self.weights["w_{}".format(mode)] = self._get_param((self.in_dim, self.out_dim))

        self.modes = modes

    def _set_propagation_indices(self):  
        ## the right indices for the propagation of the weight types
        self.node_indices = dict()
        self.rel_indices = dict()

        # edge dropout
        edge_index_masked = self.edge_index[:, self.edge_mask]
        edge_type_masked = self.edge_type[self.edge_mask]

        # self-loop
        self_node_index = torch.stack(
            [torch.arange(self.num_entities), torch.arange(self.num_entities)]
        ).to(self.device)
        self_rel_index = torch.full(
            (self.num_entities,), self.num_relations, dtype=torch.long).to(self.device)
        # self-edge dropout
        self_node_index_masked = self_node_index[:, self.self_edge_mask]
        self_rel_index_masked = self_rel_index[self.self_edge_mask]

        if self.propagation_type == "single":    
            self.node_indices[""] = torch.cat(
                [edge_index_masked, self_node_index], dim=1)            
            self.rel_indices[""] = torch.cat(
                [edge_type_masked, self_rel_index], dim=0)
            self.self_edge_weight = False

        elif self.propagation_type == "single_with_self_edge_weight": 
            self.node_indices[""] = edge_index_masked
            self.rel_indices[""] = edge_type_masked

        elif self.propagation_type == "direction":
            num_edges = edge_index_masked.size(1) // 2

            self.node_indices["in"] = edge_index_masked[:, :num_edges]
            self.node_indices["out"] = edge_index_masked[:, num_edges:]

            self.rel_indices["in"] = edge_type_masked[:num_edges]
            self.rel_indices["out"] = edge_type_masked[num_edges:]

        elif self.propagation_type in ["per_relation_basis", "per_relation_block"]:
            for rel in range(self.num_relations):
                # get the indices of the edges with the specific relation type
                rel_index = (edge_index_masked == rel).nonzero(as_tuple=True)[0]
                self.node_indices[str(rel)] = edge_index_masked[:, rel_index]
                self.rel_indices[str(rel)] = edge_type_masked[rel_index] 
        else:
            raise NotImplementedError

        if self.self_edge_weight:    
            # creates an edge index to itself for every node, e.g. 3-3 or 156-156
            self.node_indices["loop"] = self_node_index_masked
            self.rel_indices["loop"] = self_rel_index_masked
    
    def _set_composition_function(self, composition_function):
        # TODO: think of smarter way --> with getattr? and definition below
        # TODO: add the others for the other models
        if composition_function == "neighbor":
            def neighbor(h_i, h_j, h_r):
                return h_j
            self.composition = neighbor
        elif composition_function == "sub":
            def sub(h_i, h_j, h_r):
                return h_j-h_r   
            self.composition = sub
        elif composition_function == "mult":
            def mult(h_i, h_j, h_r):
                return h_j*h_r   
            self.composition = mult
        elif composition_function == "ccorr":
            def ccorr(h_i, h_j, h_r):
                return torch.irfft(
                    com_mult(conj(torch.rfft(h_j, 1)), torch.rfft(h_r, 1)),
                    1, 
                    signal_sizes=(h_r.shape[-1],)
                )  
            self.composition = ccorr
        else:
             raise NotImplementedError
       # self.composition = foo

class RgnnModel(KgeBase):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        dim: int,
        init_for_load_only=False,
    ):
        super().__init__(config, dataset, configuration_key)

        # self.edge_index, self.edge_type = self.dataset.construct_adjacency()
        edge_index = self.dataset.index("edge_index")
        # edge_type = self.dataset.index("edge_type")
        num_edges = edge_index.size(1)
        num_entities = self.dataset.num_entities()
        dim = dim
        num_layers = self.get_option("num_layers")
        device = self.config.get("job.device")
        emb_propagation_dropout = self.get_option("emb_propagation_dropout")
        # self_loop_dropout = self.get_option("self_loop_dropout")
        emb_entity_dropout = self.get_option("emb_entity_dropout")
        bias = self.get_option("bias")
        composition = self.get_option("composition")
        propagation = self.get_option("propagation")
        use_edge_norm = self.get_option("edge_norm")
        num_blocks_or_bases = self.get_option("num_blocks_or_bases")
        rel_transformation = self.get_option("rel_transformation")
        # activation function
        try:
            activation = getattr(torch.nn.functional, self.get_option("activation"))
        except Exception as e:
            raise ValueError(
                "invalid activation function: {}".format(
                    self.get_option("activation")
                )
            ) from e

        # weight initialisation
        try:
            weight_init = getattr(torch.nn.init, self.get_option("weight_init"))
        except Exception as e:
            raise ValueError(#
                "invalid weight initialisaiton: {}".format(
                    self.get_option("weight_init")
                )
            ) from e

        # edge and self-edge dropout
        # edge_dropout: make sure that the respective inverses get dropped as
        # well
        # TODO: compatibility in case of no inverse relations
        edge_dropout = self.get_option("edge_dropout")
        edge_mask_one_direction = torch.rand(int(num_edges/2)) > edge_dropout
        edge_mask = torch.cat([edge_mask_one_direction, edge_mask_one_direction])
        
        self_edge_dropout = self.get_option("self_edge_dropout")
        self_edge_mask = torch.rand(num_entities) > self_edge_dropout
        

        self.gnn_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            # TODO in_dim and out dim: factor out? TEST THIS
            if i == 0:
                self.in_dim = dim # take it from the embedders 
            else:
                # take the output dim of the previous layer as input dim
                self.in_dim = self.out_dim
                # try to read from config, if not take out_dim from last layer 
                # TODO: muss das nicht aus Prinzip immer das out_dim des
                # letzten sein?   
                # try:
                #     in_dim_key = str(i + 1) + "_in_dim"
                #     self.in_dim = self.get_option(in_dim_key)
                # except KeyError:
                #     self.in_dim = self.out_dim
            try:
                # try to read from config, if not there take same dim as in_dim
                out_dim_key = str(i + 1) + "_out_dim"
                self.out_dim = self.get_option(out_dim_key)
            except KeyError:
                self.out_dim = in_dim
            
            # in case of relation basis decomposition, get the number of basis
            # vectors of the layer
            try:
                num_rel_bases = self.get_option(str(i + 1) + "_num_rel_bases")
            except KeyError:
                num_rel_bases = -1


            # TODO: kann man das besser verpacken? oder doch die config einfach
            # mitgeben?
            self.gnn_layers.append(
                RgnnLayer(
                    self.config, self.dataset, use_edge_norm, 
                    emb_propagation_dropout, emb_entity_dropout, 
                    bias, propagation, self.in_dim, self.out_dim, composition, 
                    activation, rel_transformation, edge_mask, self_edge_mask,
                    weight_init, num_rel_bases, num_blocks_or_bases     
                )
            )
            

    def forward(self, x, r):
        for i in range(len(self.gnn_layers)):
            x, r = self.gnn_layers[i](x, r)

        return x, r[:self.dataset.num_relations()]


class RgnnEncoder(KgeRgnnEncoder):
    """Computes Relational Graph Neural Network Embeddings on top of the entity
    and relation embedders."""

    def __init__(
        self, config, dataset, configuration_key, entity_embedder, 
        relation_embedder, init_for_load_only=False
    ):
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only
        )
        self.device = self.config.get("job.device")

        self.entity_embedder = entity_embedder
        self.relation_embedder = relation_embedder
        #x = self.entity_embedder.embed_all().to(self.device) 
        #r = self.relation_embedder.embed_all().to(self.device) 

        dim = self.entity_embedder.dim
        self.rgnn = RgnnModel(
            self.config, self.dataset, self.configuration_key, dim
        )

        # self.rel_transformation = self.config.get("rel_transformation")

        #self._stored_batch_embeddings = None
        

    # def _reset_batch_embeddings(self):
    #     self._stored_batch_embeddings = None

    # def prepare_job(self, job: "Job", **kwargs):
    #     from kge.job import TrainingJob

    #     super().prepare_job(job, **kwargs)
    #     if isinstance(job, TrainingJob):
    #         job.pre_batch_hooks.append(lambda job: self._reset_batch_embeddings())

    def _run_rgnn(self):

        ent_emb, rel_emb = self.rgnn.forward(
            self.entity_embedder.embed_all(), 
            self.relation_embedder.embed_all()
        )

        return ent_emb, rel_emb

    def encode_spo(self, s, p, o, entity_subset=None):
        ent_embeddings, rel_embeddings = self._run_rgnn()
        s, p, o = self._select_embeddings(ent_embeddings, rel_embeddings, s, p, o)

        if isinstance(entity_subset, torch.Tensor):
            ent_sub = torch.index_select(ent_embeddings, 0, entity_subset.long())
            return s, p, o, ent_sub
        elif entity_subset == "all":
            ent_sub = ent_embeddings
            return s, p, o, ent_sub
        else:
            return s, p, o

    def _select_embeddings(self, entity_embeddings, relation_embeddings, s, p, o):
        if s is not None:
            s = torch.index_select(entity_embeddings, 0, s.long())
        else:
            s = entity_embeddings
        if p is not None:
            p = torch.index_select(relation_embeddings, 0, p.long())
        else:
            p = relation_embeddings
        if o is not None:
            o = torch.index_select(entity_embeddings, 0, o.long())
        else:
            o = entity_embeddings
        
        return s, p, o


# ---- Helper Functions ---- #

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mul(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.scatter_mul(src, index, dim, out, dim_size)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:

    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.floor_divide_(count)
    return out


def scatter_min(
        src: torch.Tensor, index: torch.Tensor, dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_min(src, index, dim, out, dim_size)


def scatter_max(
        src: torch.Tensor, index: torch.Tensor, dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_max(src, index, dim, out, dim_size)


def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
            out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = "sum") -> torch.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == 'mul':
        return scatter_mul(src, index, dim, out, dim_size)
    elif reduce == 'mean':
        return scatter_mean(src, index, dim, out, dim_size)
    elif reduce == 'min':
        return scatter_min(src, index, dim, out, dim_size)[0]
    elif reduce == 'max':
        return scatter_max(src, index, dim, out, dim_size)[0]
    else:
        raise ValueError

def scatter_(name, src, index, dim_size=None):
	r"""Aggregates all values from the :attr:`src` tensor at the indices
	specified in the :attr:`index` tensor along the first dimension.
	If multiple indices reference the same location, their contributions
	are aggregated according to :attr:`name` (either :obj:`"add"`,
	:obj:`"mean"` or :obj:`"max"`).

	Args:
		name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
			:obj:`"max"`).
		src (Tensor): The source tensor.
		index (LongTensor): The indices of elements to scatter.
		dim_size (int, optional): Automatically create output tensor with size
			:attr:`dim_size` in the first dimension. If set to :attr:`None`, a
			minimal sized output tensor is returned. (default: :obj:`None`)

	:rtype: :class:`Tensor`
	"""
	if name == 'add': name = 'sum'
	assert name in ['sum', 'mean', 'max']
	out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=name)
	return out[0] if isinstance(out, tuple) else out