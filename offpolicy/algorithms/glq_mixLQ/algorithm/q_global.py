import torch
import torch.nn as nn
import torch.nn.functional as F
from offpolicy.utils.util import init, to_torch


class q_global(nn.Module):
    """
    Computes total Q values given agent q values and global states.
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param num_agents: (int) number of agents in env
    :param cent_obs_dim: (int) dimension of the centralized state
    :param device: (torch.Device) torch device on which to do computations.
    :param multidiscrete_list: (list) list of each action dimension if action space is multidiscrete
    """

    def __init__(self, args, num_agents, cent_obs_dim, act_dim, device, multidiscrete_list=None):
        super(q_global, self).__init__()
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = num_agents
        self.share_hyper_network = args.share_hyper_network
        self.use_same_share_obs = args.use_same_share_obs
        self.cent_obs_dim = cent_obs_dim
        self.act_dim = act_dim
        self._use_orthogonal = args.use_orthogonal

        if multidiscrete_list:
            self.num_q_inps = sum(multidiscrete_list)
        else:
            self.num_q_inps = self.num_agents
        # dimension of the hidden layer of the mixing net
        self.hidden_layer_dim = args.mixer_hidden_dim
        # dimension of the hidden layer of each hypernet
        self.hypernet_hidden_dim = args.hypernet_hidden_dim

        self.hypernet_output_dim = self.hidden_layer_dim
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        self.hypernet_num = 1 if self.share_hyper_network else self.num_q_inps

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        # hypernets output the weight and bias for the 2 layer MLP which takes in the state and agent Qs and outputs Q_tot
        if args.hypernet_layers == 1:
            # each hypernet only has 1 layer to output the weights
            # hyper_w1 outputs weight matrix which is of dimension (hidden_layer_dim x N)
            self.hyper_w1 = nn.Sequential(*[init_(nn.Linear(self.cent_obs_dim, self.hypernet_output_dim * self.num_q_inps))
                                            for _ in range(self.hypernet_num)])
            # hyper_w2 outputs weight matrix which is of dimension (1 x hidden_layer_dim)
            self.hyper_w2 = nn.Sequential(*[init_(nn.Linear(self.cent_obs_dim, self.hypernet_output_dim))
                                            for _ in range(self.hypernet_num)])
            # hyper_b1 outputs bias vector of dimension (1 x hidden_layer_dim)
            self.hyper_b1 = nn.Sequential(*[init_(nn.Linear(self.cent_obs_dim, self.hypernet_output_dim))
                                            for _ in range(self.hypernet_num)])
            # hyper_b2 outputs bias vector of dimension (1 x 1)
            self.hyper_b2 = nn.Sequential(*[init_(nn.Linear(self.cent_obs_dim, 1))
                                            for _ in range(self.hypernet_num)])
        elif args.hypernet_layers == 2:
            # 2 layer hypernets: output dimensions are same as above case
            self.hyper_w1 = nn.Sequential(*[nn.Sequential(
                init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
                nn.ReLU(),
                init_(nn.Linear(self.hypernet_hidden_dim, self.hypernet_output_dim * self.num_q_inps)))
                for _ in range(self.hypernet_num)])
            self.hyper_w2 = nn.Sequential(*[nn.Sequential(
                init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
                nn.ReLU(),
                init_(nn.Linear(self.hypernet_hidden_dim, self.hypernet_output_dim)))
                for _ in range(self.hypernet_num)])
            # hyper_b1 outputs bias vector of dimension (1 x hidden_layer_dim)
            self.hyper_b1 = nn.Sequential(*[nn.Sequential(
                                            init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
                                            nn.ReLU(),
                                            init_(nn.Linear(self.hypernet_hidden_dim, self.hypernet_output_dim)))
                                            for _ in range(self.hypernet_num)])
            # hyper_b2 outputs bias vector of dimension (1 x 1)
            self.hyper_b2 = nn.Sequential(*[nn.Sequential(
                                            init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
                                            nn.ReLU(),
                                            init_(nn.Linear(self.hypernet_hidden_dim, 1)))
                                            for _ in range(self.hypernet_num)])
        self.to(device)

    def forward(self, agent_q_individual, states):
        """
         Computes Q_tot using the individual agent q values and global state.
         :param agent_q_inps: (torch.Tensor) individual agent q values
         :param states: (torch.Tensor) state input to the hypernetworks.
         :return Q_tot: (torch.Tensor) computed Q_tot values
         """
        agent_q_individual = to_torch(agent_q_individual).to(**self.tpdv)
        states = to_torch(states).to(**self.tpdv)

        batch_size = agent_q_individual.size(1)
        if self.use_same_share_obs:
            states = states.view(-1, batch_size, 1, self.cent_obs_dim).float()
        else:
            states = states.view(-1, batch_size, self.num_q_inps, self.cent_obs_dim).float()
        agent_q_individual = agent_q_individual.view(-1, batch_size, 1, 1, self.num_q_inps)
        agent_q_inps = torch.cat([agent_q_individual[..., [n] + list(range(0, n)) + list(range(n + 1, self.num_q_inps))]
                                            for n in range(self.num_q_inps)], dim=-3)
        # 对hyper网络进行前向传播得到global网络参数
        if self.share_hyper_network:
            w1 = torch.abs(self.hyper_w1(states))
            b1 = self.hyper_b1(states)
            w2 = torch.abs(self.hyper_w2(states))
            b2 = self.hyper_b2(states)
        else:
            w1 = []
            b1 = []
            for i, hyper_w1 in enumerate(self.hyper_w1.children()):
                w1.append(torch.abs(hyper_w1(states[:,:,0 if self.use_same_share_obs else i,:])))
            for i, hyper_b1 in enumerate(self.hyper_b1.children()):
                b1.append(hyper_b1(states[:,:,0 if self.use_same_share_obs else i,:]))
            w2 = []
            b2 = []
            for i, hyper_w2 in enumerate(self.hyper_w2.children()):
                w2.append(torch.abs(hyper_w2(states[:,:,0 if self.use_same_share_obs else i,:])))
            for i, hyper_b2 in enumerate(self.hyper_b2.children()):
                b2.append(hyper_b2(states[:,:,0 if self.use_same_share_obs else i,:]))
            w1 = torch.cat(w1, dim=-1)
            b1 = torch.cat(b1, dim=-1)
            w2 = torch.cat(w2, dim=-1)
            b2 = torch.cat(b2, dim=-1)
        # 对global网络参数预处理
        if self.use_same_share_obs and self.share_hyper_network:
            w1=w1.repeat(1,1,3,1)
            w2=w2.repeat(1,1,3,1)
            b1=b1.repeat(1,1,3,1)
            b2=b2.repeat(1,1,3,1)
        # 参数各个维度分别代表(episode总step数，批量，智能体个数，输入维度（输入一个q），输出维度)
        w1 = w1.view(-1, batch_size, self.num_q_inps, self.num_q_inps, self.hidden_layer_dim)
        b1 = b1.view(-1, batch_size, self.num_q_inps, 1, self.hidden_layer_dim)
        w2 = w2.view(-1, batch_size, self.num_q_inps, self.hidden_layer_dim, 1)
        b2 = b2.view(-1, batch_size, self.num_q_inps, 1, 1)
        # 对global网络前向传播
        hidden_layer = F.elu(torch.matmul(agent_q_inps, w1) + b1)
        out = torch.matmul(hidden_layer, w2) + b2
        q_global = out.view(-1, batch_size, self.num_q_inps)
        return q_global
