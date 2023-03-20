import torch
import torch.nn as nn
import torch.nn.functional as F
from offpolicy.utils.util import init, to_torch

class QMixer(nn.Module):
    """
    Computes total Q values given agent q values and global states.
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param num_agents: (int) number of agents in env
    :param cent_obs_dim: (int) dimension of the centralized state
    :param device: (torch.Device) torch device on which to do computations.
    :param multidiscrete_list: (list) list of each action dimension if action space is multidiscrete
    """

    def __init__(self, args, num_agents, cent_obs_dim, device, state_resort_orders, multidiscrete_list=None):
        super(QMixer, self).__init__()
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = num_agents
        self.cent_obs_dim = cent_obs_dim
        self._use_orthogonal = args.use_orthogonal

        # dimension of the hidden layer of the mixing net
        self.hidden_layer_dim = args.mixer_hidden_dim
        # dimension of the hidden layer of each hypernet
        self.hypernet_hidden_dim = args.hypernet_hidden_dim
        self.hypernet_output_dim = self.hidden_layer_dim

        # multi-head QMix
        self.resort_q = args.resort_q
        self.hypernet_num = 1 if args.share_hyper_network else args.num_perspective
        # self.num_perspective_in_state only represents the number of perspectives in the state
        self.num_perspective_in_state = 1 if args.use_same_share_obs else args.num_perspective  
        self.state_resort_orders = range(self.num_agents) if args.use_same_share_obs else state_resort_orders
        self.share_hyper_network = args.share_hyper_network
        self.use_same_share_obs = args.use_same_share_obs

        if multidiscrete_list:
            self.num_mixer_q_inps = sum(multidiscrete_list)
        else:
            self.num_mixer_q_inps = self.num_agents

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # hypernets output the weight and bias for the 2 layer MLP which takes in the state and agent Qs and outputs Q_tot
        if args.hypernet_layers == 1:
            # each hypernet only has 1 layer to output the weights
            # hyper_w1 outputs weight matrix which is of dimension (hidden_layer_dim x N)
            self.hyper_w1 = nn.Sequential(
                *[init_(nn.Linear(self.cent_obs_dim, self.hypernet_output_dim * self.num_mixer_q_inps))
                  for _ in range(self.hypernet_num)])
            # hyper_w2 outputs weight matrix which is of dimension (1 x hidden_layer_dim)
            self.hyper_w2 = nn.Sequential(*[init_(nn.Linear(self.cent_obs_dim, self.hypernet_output_dim))
                                            for _ in range(self.hypernet_num)])
        elif args.hypernet_layers == 2:
            # 2 layer hypernets: output dimensions are same as above case
            self.hyper_w1 = nn.Sequential(*[nn.Sequential(
                init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
                nn.ReLU(),
                init_(nn.Linear(self.hypernet_hidden_dim, self.hypernet_output_dim * self.num_mixer_q_inps)))
                for _ in range(self.hypernet_num)])
            self.hyper_w2 = nn.Sequential(*[nn.Sequential(
                init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
                nn.ReLU(),
                init_(nn.Linear(self.hypernet_hidden_dim, self.hypernet_output_dim)))
                for _ in range(self.hypernet_num)])
        # hyper_b1 outputs bias vector of dimension (1 x hidden_layer_dim)
        self.hyper_b1 = nn.Sequential(*[init_(nn.Linear(self.cent_obs_dim, self.hidden_layer_dim))
                                        for _ in range(self.hypernet_num)])
        # hyper_b2 outputs bias vector of dimension (1 x 1)
        self.hyper_b2 = nn.Sequential(*[nn.Sequential(
            init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
            nn.ReLU(),
            init_(nn.Linear(self.hypernet_hidden_dim, 1)))
            for _ in range(self.hypernet_num)])
        self.to(device)

    def forward(self, agent_q_inps, states):
        """
         Computes Q_tot using the individual agent q values and global state.
         :param agent_q_inps: (torch.Tensor) individual agent q values
         :param states: (torch.Tensor) state input to the hypernetworks.
         :return Q_tot: (torch.Tensor) computed Q_tot values
         """
        agent_q_inps = to_torch(agent_q_inps).to(**self.tpdv)
        states = to_torch(states).to(**self.tpdv)

        batch_size = agent_q_inps.size(1)
        states = states.view(-1, batch_size, self.num_perspective_in_state, self.cent_obs_dim).float()
        agent_q_inps = agent_q_inps.view(-1, batch_size, 1, 1, self.num_mixer_q_inps)
        if not self.resort_q or self.use_same_share_obs:
            agent_q_inps = agent_q_inps.repeat(1, 1,
                                               self.num_perspective_in_state if self.share_hyper_network else self.hypernet_num,
                                               1, 1)
        else:
            agent_q_inps = torch.cat([agent_q_inps[..., order] for order in self.state_resort_orders], dim=-3)
        if self.share_hyper_network:
            w1 = torch.abs(self.hyper_w1(states))
            b1 = self.hyper_b1(states)
            w2 = torch.abs(self.hyper_w2(states))
            b2 = self.hyper_b2(states)
            w1 = w1.view(-1, batch_size, self.num_perspective_in_state, self.num_mixer_q_inps, self.hidden_layer_dim)
            b1 = b1.view(-1, batch_size, self.num_perspective_in_state, 1, self.hidden_layer_dim)
            w2 = w2.view(-1, batch_size, self.num_perspective_in_state, self.hidden_layer_dim, 1)
            b2 = b2.view(-1, batch_size, self.num_perspective_in_state, 1, 1)
        else:
            w1 = []
            b1 = []
            for i, hyper_w1 in enumerate(self.hyper_w1.children()):
                w1.append(torch.abs(hyper_w1(states[:,:,:])))
            for i, hyper_b1 in enumerate(self.hyper_b1.children()):
                b1.append(hyper_b1(states[:,:,0 if self.use_same_share_obs else i,:]))
            w2 = []
            b2 = []
            for i, hyper_w2 in enumerate(self.hyper_w2.children()):
                w2.append(torch.abs(hyper_w2(states[:,:,:])))
            for i, hyper_b2 in enumerate(self.hyper_b2.children()):
                b2.append(hyper_b2(states[:,:,0 if self.use_same_share_obs else i,:]))
            w1 = torch.cat(w1, dim=-1)
            b1 = torch.cat(b1, dim=-1)
            w2 = torch.cat(w2, dim=-1)
            b2 = torch.cat(b2, dim=-1)
            w1 = w1.view(-1, batch_size, self.hypernet_num, self.num_mixer_q_inps, self.hidden_layer_dim)
            b1 = b1.view(-1, batch_size, self.hypernet_num, 1, self.hidden_layer_dim)
            w2 = w2.view(-1, batch_size, self.hypernet_num, self.hidden_layer_dim, 1)
            b2 = b2.view(-1, batch_size, self.hypernet_num, 1, 1)

        hidden_layer = F.elu(torch.matmul(agent_q_inps, w1) + b1)
        out = torch.matmul(hidden_layer, w2) + b2
        q_tot = out.view(-1, batch_size, self.num_perspective_in_state if self.share_hyper_network else self.hypernet_num)
        return q_tot
