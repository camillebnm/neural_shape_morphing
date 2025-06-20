# coding: utf-8

from collections import OrderedDict
import os.path as osp
import torch
from torch import nn
import numpy as np
from src.diff_operators import rotationnel, gradient, jacobien, divergence


@torch.no_grad()
def sine_init(m, w0):
    if hasattr(m, 'weight'):
        num_input = m.weight.size(-1)
        m.weight.uniform_(-np.sqrt(6 / num_input) / w0,
                          np.sqrt(6 / num_input) / w0)


@torch.no_grad()
def first_layer_sine_init(m):
    if hasattr(m, 'weight'):
        num_input = m.weight.size(-1)
        m.weight.uniform_(-1 / num_input, 1 / num_input)


def siren_v1_to_v2(model_in, check_equals=False):
    """Converts the models trained using the old class to the new format.

    Parameters
    ----------
    model_in: OrderedDict
        Model trained by our old SIREN version (Sitzmann code).

    check_equals: boolean, optional
        Whether to check if the converted models weight match. By default this
        is False.

    Returns
    -------
    model_out: OrderedDict
        The input model converted to a format recognizable by our version of
        SIREN.

    divergences: list[tuple[str, str]]
        If `check_equals` is True, then this list contains the keys where the
        original and converted model dictionaries are not equal. Else, this is
        an empty list.

    See Also
    --------
    `model.SIREN`
    """
    model_out = OrderedDict()
    for k, v in model_in.items():
        model_out[k[4:]] = v

    divergences = []
    if check_equals:
        for k in model_in.keys():
            test = model_in[k] == model_out[k[4:]]
            if test.sum().item() != test.numel():
                divergences.append((k, k[4:]))

    return model_out, divergences


def from_state_dict(weights: OrderedDict, device: str = "cpu", w0=1, ww=None):
    """Builds a SIREN network with the topology and weights in `weights`.

    Parameters
    ----------
    weights: OrderedDict
        The input state_dict to use as reference.

    device: str, optional
        Device to load the weights. Default value is cpu.

    w0: number, optional
        Frequency parameter for the first layer. Default value is 1.

    ww: number, optional
        Frequency parameter for the intermediate layers. Default value is None,
        we will assume that ww = w0 in this case

    Returns
    -------
    model: nifm.model.SIREN
        The NN model mirroring `weights` topology.

    upgrade_to_v2: boolean
        If `weights` was from an older version of SIREN, we convert them to our
        format and set this to `True`, signaling this fact.
    """
    n_layers = len(weights) // 2
    hidden_layer_config = [None] * (n_layers - 1)
    keys = list(weights.keys())

    bias_keys = [k for k in keys if "bias" in k] #"biais"
    i = 0
    while i < (n_layers - 1):
        k = bias_keys[i]
        hidden_layer_config[i] = weights[k].shape[0]
        i += 1

    n_in_features = weights[keys[0]].shape[1]
    n_out_features = weights[keys[-1]].shape[0]
    model = SIREN(
        n_in_features=n_in_features,
        n_out_features=n_out_features,
        hidden_layer_config=hidden_layer_config,
        w0=w0, ww=ww, delay_init=True
    )

    # Loads the weights. Converts to version 2 if they are from the old version
    # of SIREN.
    upgrade_to_v2 = False
    try:
        model.load_state_dict(weights)
    except RuntimeError:
        print("Found weights from old version of SIREN. Converting to v2.")
        new_weights, diff = siren_v1_to_v2(weights, True)
        model.load_state_dict(new_weights)
        upgrade_to_v2 = True

    return model, upgrade_to_v2

def from_state_dict_lip(weights: OrderedDict, device: str = "cpu", w0=1, ww=None):
    """Builds a SIREN network with the topology and weights in `weights`.

    Parameters
    ----------
    weights: OrderedDict
        The input state_dict to use as reference.

    device: str, optional
        Device to load the weights. Default value is cpu.

    w0: number, optional
        Frequency parameter for the first layer. Default value is 1.

    ww: number, optional
        Frequency parameter for the intermediate layers. Default value is None,
        we will assume that ww = w0 in this case

    Returns
    -------
    model: nifm.model.SIREN
        The NN model mirroring `weights` topology.

    upgrade_to_v2: boolean
        If `weights` was from an older version of SIREN, we convert them to our
        format and set this to `True`, signaling this fact.
    """
    n_layers = len(weights) // 3
    hidden_layer_config = [None] * (n_layers - 1)
    keys = list(weights.keys())

    bias_keys = [k for k in keys if "b" in k] #"biais"
    i = 0
    print(n_layers)
    print(bias_keys)
    while i < (n_layers - 1):
        k = bias_keys[i]
        hidden_layer_config[i] = weights[k].shape[0]
        i += 1

    n_in_features = weights[keys[0]].shape[1]
    n_out_features = weights[bias_keys[-1]].shape[0]

    print(n_out_features)
    print(n_in_features)
    print(hidden_layer_config)
    
    model = LipschitzMLP(
        n_in_features=n_in_features,
        n_out_features=n_out_features,
        hidden_layer_config=hidden_layer_config,
        w0=w0
    )

    # Loads the weights. Converts to version 2 if they are from the old version
    # of SIREN.
    upgrade_to_v2 = False
    try:
        model.load_state_dict(weights)
    except RuntimeError:
        print("Found weights from old version of SIREN. Converting to v2.")
        new_weights, diff = siren_v1_to_v2(weights, True)
        model.load_state_dict(new_weights)
        upgrade_to_v2 = True

    return model, upgrade_to_v2


def from_pth(path, device="cpu", w0=1, ww=None, lip=False):
    """Builds a SIREN given a weights file.

    Parameters
    ----------
    path: str
        Path to the pth file.

    device: str, optional
        Device to load the weights. Default value is cpu.

    w0: number, optional
        Frequency parameter for the first layer. Default value is 1.

    ww: number, optional
        Frequency parameter for the intermediate layers. Default value is None,
        we will assume that ww = w0 in this case.

    Returns
    -------
    model: torch.nn.Module
        The resulting model.

    Raises
    ------
    FileNotFoundError if `path` points to a non-existing file.
    
    """
    if not osp.exists(path):
        raise FileNotFoundError(f"Weights file not found at \"{path}\"")
    if lip : 
        

    
        weights = torch.load(path, map_location=torch.device(device))
        model, upgraded_to_v2 = from_state_dict_lip(
            weights, device=device, w0=w0, ww=ww
        )
    else : 
        weights = torch.load(path, map_location=torch.device(device))
        model, upgraded_to_v2 = from_state_dict(
            weights, device=device, w0=w0, ww=ww
        )
        if upgraded_to_v2:
            torch.save(model.state_dict(), path.split(".")[0] + "_v2.pth")    

    return model.to(device=device)


class SineLayer(nn.Module):
    """A Sine non-linearity layer.
    """
    def __init__(self, w0=30):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

    def __repr__(self):
        return f"SineLayer(w0={self.w0})"


class SIREN(nn.Module):
    """SIREN Module

    Parameters
    ----------
    n_in_features: int
        Number of input features.

    n_out_features: int
        Number of output features.

    hidden_layer_config: list[int], optional
        Number of neurons at each hidden layer of the network. The model will
        have `len(hidden_layer_config)` hidden layers. Only used in during
        model training. Default value is None.

    w0: number, optional
        Frequency multiplier for the Sine layers. Only useful for training the
        model. Default value is 1.

    ww: number, optional
        Frequency multiplier for the hidden Sine layers. Only useful for
        training the model. Default value is None.

    delay_init: boolean, optional
        Indicates if we should perform the weight initialization or not.
        Default value is False, meaning that we perform the weight
        initialization as usual. This is useful if we will load the weights of
        a pre-trained network, in this case, initializing the weights does not
        make sense, since they will be overwritten.

    References
    ----------
    [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
    & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
    Activation Functions. ArXiv. http://arxiv.org/abs/2006.09661
    """
    def __init__(self, n_in_features, n_out_features, hidden_layer_config=[],
                 w0=1, ww=None, delay_init=False):
        super(SIREN, self).__init__()
        self.in_features = n_in_features
        self.out_features = n_out_features
        self.w0 = w0
        if ww is None:
            self.ww = w0
        else:
            self.ww = ww

        net = []
        net.append(nn.Sequential(
            nn.Linear(n_in_features, hidden_layer_config[0]),
            SineLayer(self.w0)
        ))

        for i in range(1, len(hidden_layer_config)):
            net.append(nn.Sequential(
                nn.Linear(hidden_layer_config[i-1], hidden_layer_config[i]),
                SineLayer(self.ww)
            ))

        net.append(nn.Sequential(
            nn.Linear(hidden_layer_config[-1], n_out_features),
        ))

        self.net = nn.Sequential(*net)
        if not delay_init:
            self.reset_weights()

    def forward(self, x, omegas=dict(),sdf=None):
        """Forward pass of the model.

        Parameters
        ----------
        x: torch.Tensor
            The model input containing of size Nx3

        omegas: dict[int=>float]
            The omega values to apply to the input. Must be a dict with the
             coordinate index as key, and new omega value for said coordinate
             as value.

        Returns
        -------
        dict
            Dictionary of tensors with the input coordinates under 'model_in'
            and the model output under 'model_out'.
        """

        # WARNING: it is changing the values of the time coords, thus, we need
        # to be careful in the loss function conditions
        if omegas:
            for k, v in omegas.items():
                if k < 0 or k > x.shape[1]:
                    continue
                x[..., k] = v * x[..., k] / self.w0

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = x.clone().detach().requires_grad_(True)
        coords = coords_org
        y = self.net(coords)
        return {"model_in": coords_org, "model_out": y}

    def reset_weights(self):
        """Resets the weights of the network using Sitzmann et al. (2020).

        Returns
        -------
        self: nise.model.SIREN
            The network.
        """
        self.net[0].apply(first_layer_sine_init)
        self.net[1:].apply(lambda module: sine_init(module, self.ww))

    def update_omegas(self, w0=1, ww=None):
        """Updates the omega values for all layers except the last one.

        Note that this updates the w0 and ww instance attributes.

        Parameters
        ----------
        w0: number, optional
            The new omega_0 value to assume. By default is 1.

        ww: number, optional
            The new omega_w value to assume. By default is None, meaning
            that `ww` wil be set to `w0`.
        """
        if ww is None:
            ww = w0

        # Updating the state_dict weights and biases.
        my_sd = self.state_dict()
        keys = list(my_sd.keys())
        for k in keys[:-2]:
            my_sd[k] = my_sd[k] * (self.w0 / w0)

        self.load_state_dict(my_sd)
        self.w0 = w0
        self.ww = ww

        # Updating the omega values of the SineLayer instances
        self.net[0][1].w0 = w0
        for i in range(1, len(self.net)-1):
            self.net[i][1].w0 = ww

        return self

    def from_pretrained_initial_condition(self, other: OrderedDict):
        """Neural network initialization given a pretrained network.

        This method assumes that the network defined by `self` is as deep as
        `other`, and each layer is at least as wide as `other` as well. If the
        depth and `self, or the width of `other` is larger than `self`, the
        method will abort.

        Let us consider `other`'s weights to be `B`, while `self`'s weights are
        `A`. Our initialization assigns:
        $A_1 = ( B_1 0; f1 f2 )$ and
        $A_i = ( B_i 0; 0  0  ), i > 1$

        where $f1$ and $f2$ are weight values initilized as proposed by [1].

        The biases are defined as the column
        vector:
        $ a_i = ( b_i 0 )^T $

        Parameters
        ----------
        other: OrderedDict
            The state_dict to use as reference. Note that it must have the same
            depth as `model.state_dict()`, and the first layer weights of this
            state_dict must have shape [N, D-1], where `D` is the number of
            columns of model.state_dict()["net.0.0.weight"].

        Returns
        -------
        model: niif.model.SIREN
            The initialized model

        References
        ----------
        [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
        & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
        Activation Functions. ArXiv. http://arxiv.org/abs/2006.09661
        """
        depth_other = len(other) // 2
        depth_us = len(self.state_dict()) // 2
        if depth_other != depth_us:
            raise ValueError("Number of layers does not match.")

        try:
            first_layer = other["net.0.0.weight"]
        except KeyError:
            raise ValueError("Invalid state_dict provided."
                             " Convert it to v2 first")

        if first_layer.shape[1] != self.net[0][0].weight.shape[1] - 1:
            raise ValueError(
                "Invalid first-layer size on the reference weights."
                f" Is {first_layer.shape[1]}, should be"
                f" {self.net[0][0].weights.shape[1] - 1}."
            )

        my_sd = self.state_dict()
        for k, v in my_sd.items():
            if v.shape[0] < other[k].shape[0]:
                raise AttributeError(
                    "The input layer has more rows than ours. Ensure that they"
                    " either match, or ours is taller than the input."
                )
            if v.ndim > 1:
                if v.shape[1] < other[k].shape[1]:
                    raise AttributeError(
                        "The input layer has more columns than ours. Ensure"
                        " that they either match, or ours is wider than the"
                        " input."
                    )

        # Appending new column to input weights (all zeroes)
        new_first_layer = torch.cat((
            first_layer, torch.zeros_like(first_layer[..., 0].unsqueeze(-1))
        ), dim=-1)

        flh, flw = new_first_layer.shape
        keys = list(my_sd.keys())
        # Ensuring that the input layer weights are all zeroes.
        # my_sd[keys[0]] = torch.zeros_like(my_sd[keys[0]])
        my_sd[keys[0]].uniform_(-1/4, 1/4)
        my_sd[keys[0]][:flh, :flw] = new_first_layer

        # Solving the last layer weights.
        # State dict keys are interleaved: weights, biases, weights, biases,...
        # The second-to-last key is the last weights tensor.
        llw = other[keys[-2]].shape[1]
        # my_sd[keys[-2]] = torch.zeros_like(my_sd[keys[-2]])
        outlayer_w = my_sd[keys[-2]].size(-1)
        m1 = np.sqrt(6.0 / outlayer_w) / self.ww
        my_sd[keys[-2]].uniform_(-m1, m1)
        my_sd[keys[-2]][:, :llw] = other[keys[-2]]

        # Handling the intermediate layer weights.
        for k in keys[2:-2:2]:
            hh, hw = other[k].shape
            z = torch.zeros_like(my_sd[k])
            z[:hh, :hw] = other[k]
            my_sd[k] = z

        # Handling the layers biases.
        for k in keys[1::2]:
            ll = other[k].shape[0]
            z = torch.zeros_like(my_sd[k])
            z[:ll] = other[k]
            my_sd[k] = z

        self.load_state_dict(my_sd)
        return self

    def from_pretrained_initial_condition_with_noise(self, other: OrderedDict, var):
        """Neural network initialization given a pretrained network.

        This method assumes that the network defined by `self` is as deep as
        `other`, and each layer is at least as wide as `other` as well. If the
        depth and `self, or the width of `other` is larger than `self`, the
        method will abort.

        Let us consider `other`'s weights to be `B`, while `self`'s weights are
        `A`. Our initialization assigns:
        $A_1 = ( B_1 0; f1 f2 )$ and
        $A_i = ( B_i 0; 0  0  ), i > 1$

        where $f1$ and $f2$ are weight values initilized as proposed by [1].

        The biases are defined as the column
        vector:
        $ a_i = ( b_i 0 )^T $

        Parameters
        ----------
        other: OrderedDict
            The state_dict to use as reference. Note that it must have the same
            depth as `model.state_dict()`, and the first layer weights of this
            state_dict must have shape [N, D-1], where `D` is the number of
            columns of model.state_dict()["net.0.0.weight"].

        Returns
        -------
        model: niif.model.SIREN
            The initialized model

        References
        ----------
        [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
        & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
        Activation Functions. ArXiv. http://arxiv.org/abs/2006.09661
        """
        depth_other = len(other) // 2
        depth_us = len(self.state_dict()) // 2
        if depth_other != depth_us:
            raise ValueError("Number of layers does not match.")

        try:
            first_layer = other["net.0.0.weight"]
        except KeyError:
            raise ValueError("Invalid state_dict provided."
                             " Convert it to v2 first")

        if first_layer.shape[1] != self.net[0][0].weight.shape[1] - 1:
            raise ValueError(
                "Invalid first-layer size on the reference weights."
                f" Is {first_layer.shape[1]}, should be"
                f" {self.net[0][0].weights.shape[1] - 1}."
            )

        my_sd = self.state_dict()
        for k, v in my_sd.items():
            if v.shape[0] < other[k].shape[0]:
                raise AttributeError(
                    "The input layer has more rows than ours. Ensure that they"
                    " either match, or ours is taller than the input."
                )
            if v.ndim > 1:
                if v.shape[1] < other[k].shape[1]:
                    raise AttributeError(
                        "The input layer has more columns than ours. Ensure"
                        " that they either match, or ours is wider than the"
                        " input."
                    )

        # Appending new column to input weights (all zeroes)
        
        new_first_layer = torch.cat((
            first_layer, var*torch.randn((first_layer[..., 0].unsqueeze(-1))
        .shape , device="cuda") + torch.zeros_like(first_layer[..., 0].unsqueeze(-1))
        ), dim=-1)

        flh, flw = new_first_layer.shape
        keys = list(my_sd.keys())
        # Ensuring that the input layer weights are all zeroes.
        # my_sd[keys[0]] = torch.zeros_like(my_sd[keys[0]])
        my_sd[keys[0]].uniform_(-1/4, 1/4)
        my_sd[keys[0]][:flh, :flw] = new_first_layer

        # Solving the last layer weights.
        # State dict keys are interleaved: weights, biases, weights, biases,...
        # The second-to-last key is the last weights tensor.
        llw = other[keys[-2]].shape[1]
        # my_sd[keys[-2]] = torch.zeros_like(my_sd[keys[-2]])
        outlayer_w = my_sd[keys[-2]].size(-1)
        m1 = np.sqrt(6.0 / outlayer_w) / self.ww
        my_sd[keys[-2]].uniform_(-m1, m1)
        my_sd[keys[-2]][:, :llw] = other[keys[-2]]

        # Handling the intermediate layer weights.
        for k in keys[2:-2:2]:
            hh, hw = other[k].shape
            z = torch.zeros_like(my_sd[k]) + var*torch.randn(my_sd[k].shape, device="cuda")
            z[:hh, :hw] = other[k]
            my_sd[k] = z

        # Handling the layers biases.
        for k in keys[1::2]:
            ll = other[k].shape[0]
            z = torch.zeros_like(my_sd[k]) + var*torch.randn(my_sd[k].shape, device="cuda")
            z[:ll] = other[k]
            my_sd[k] = z

        self.load_state_dict(my_sd)
        return self


class HODGE_SIREN(nn.Module): 
    def __init__(self,device, n_in_features, n_out_features, hidden_layer_config=[],
                 w0=1, ww=None, alpha=0.1, delay_init=False):
        
        super(HODGE_SIREN, self).__init__()
        #The model has an attribute that is a SIREN. to acces the weights etc ... as always, use model.siren.fct instead of just model.fct
        self.siren = SIREN(n_in_features, n_out_features, hidden_layer_config,
                 w0, ww, delay_init).to(device)
        self.device = device
        self.alpha = alpha
        def f_band(x) : 
            return 1 - torch.exp(-self.alpha*x**2)
        self.f_band = f_band

    def forward(self, x,sdf) : 
      
      y = self.siren(x)
      coords_org = y["model_in"]
      y = y["model_out"]
      G = y[:,0:1]
      D = y[:,1:]
      gradG = gradient(G,coords_org)[:,:-1]
      J = jacobien(D,coords_org)
      rot = rotationnel(J,device=self.device)
      with torch.no_grad():
          ind_SDF = self.f_band(sdf)
      V = rot + gradG * ind_SDF.view(x.shape[0], 1)
      
      return {"model_in": coords_org, "model_out": V }

    def to(self,device) : 
        self.siren.to(device)
        self.device=device
        return self

    def state_dict(self) : 
        return self.siren.state_dict()
        
class DIV_SIREN(nn.Module): 
    def __init__(self,device, n_in_features, n_out_features, hidden_layer_config=[],
                 w0=1, ww=None, alpha=0.1, delay_init=False):
        
        super(DIV_SIREN, self).__init__()
        #The model has an attribute that is a SIREN. to acces the weights etc ... as always, use model.siren.fct instead of just model.fct
        self.siren = SIREN(n_in_features, n_out_features, hidden_layer_config,
                 w0, ww, delay_init).to(device)
        self.device = device

    def forward(self, x,sdf) : 
      
      y = self.siren(x)
      coords_org = y["model_in"]
      y = y["model_out"]
      J = jacobien(y,coords_org)
      rot = rotationnel(J,device=self.device)
      return {"model_in": coords_org, "model_out": rot }

    def to(self,device) : 
        self.siren.to(device)
        self.device=device
        return self

    def state_dict(self) : 
        return self.siren.state_dict()


class LipschitzMLP(nn.Module):
    """Multi-layer Perceptron with Lipschitz Regularization [1].

    Parameters
    ----------
    n_in_features: int
        Number of input features.

    n_out_features: int
        Number of output features.

    hidden_layer_config: list[int], optional
        Number of neurons at each hidden layer of the network. The model will
        have `len(hidden_layer_config)` hidden layers. Only used in during
        model training. Default value is None.

    w0: number, optional
        Frequency multiplier for the Sine layers. Only useful for training the
        model. Default value is 1.

    References
    ----------
    [1] Liu, Hsueh-Ti Derek, et al. "Learning smooth neural functions via
    lipschitz regularization." ACM SIGGRAPH 2022 Conference Proceedings. 2022
    """
    def __init__(self, n_in_features, n_out_features, hidden_layer_config=[],
                 w0=30):
        super(LipschitzMLP, self).__init__()

        def init_W(size_out, size_in):
            W = torch.randn(size_out, size_in) * torch.sqrt(torch.Tensor([2 / size_in]))
            return W

        self.w0 = w0
        sizes = hidden_layer_config
        sizes.insert(0, n_in_features)
        sizes.append(n_out_features)
        self.num_layers = len(sizes)
        self.params_W = []
        self.params_b = []
        self.params_c = []
        for ii in range(len(sizes)-1):
            W = torch.nn.Parameter(init_W(sizes[ii+1], sizes[ii]))
            b = torch.nn.Parameter(torch.zeros(sizes[ii+1]))
            c = torch.nn.Parameter(torch.max(torch.sum(torch.abs(W), axis=1)))
            self.params_W.append(W)
            self.params_b.append(b)
            self.params_c.append(c)

        self.params_W = nn.ParameterList(self.params_W)
        self.params_b = nn.ParameterList(self.params_b)
        self.params_c = nn.ParameterList(self.params_c)

    def weight_normalization(self, W, softplus_c):
        """
        Lipschitz weight normalization based on the L-infinity norm
        """
        absrowsum = torch.sum(torch.abs(W), axis=1)
        scale = torch.minimum(torch.Tensor([1.0]).cuda(), softplus_c/absrowsum)
        return W * scale[:, None]

    def get_lipschitz_loss(self):
        loss_lip = 1.0
        for ii in range(len(self.params_c)):
            c = self.params_c[ii]
            # loss_lip = loss_lip * nn.Softplus()(c)
            loss_lip = loss_lip * nn.Softplus()(c)
        return loss_lip

    def forward(self, x):
        # forward pass
        coords_org = x.clone().detach().requires_grad_(True)
        coords = coords_org
        for ii in range(len(self.params_W) - 1):
            # W, b, c = self.params_net[ii]
            W = self.params_W[ii]
            b = self.params_b[ii]
            c = self.params_c[ii]
            W = self.weight_normalization(W, nn.Softplus()(c))
            coords = nn.Tanh()(torch.matmul(coords, W.T) + b)
            # coords = nn.ReLU()(torch.matmul(coords,W.T) + b)
            # coords = nn.ELU()(torch.matmul(coords,W.T) + b)

        # final layer
        # W, b, c = self.params_net[-1]
        W = self.params_W[-1]
        b = self.params_b[-1]
        c = self.params_c[-1]
        W = self.weight_normalization(W, nn.Softplus()(c))
        out = torch.matmul(coords, W.T) + b
        return {"model_in": coords_org, "model_out": out}
