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
        self.add_other_act_to_cent = args.add_other_act_to_cent
        self.mix_local_q = args.mix_local_q
        self.use_global_mixing_network = args.use_global_mixing_network
        # if self.add_other_act_to_cent and not self.use_same_share_obs:
        self.cent_obs_dim_glq = cent_obs_dim + (num_agents-1)*act_dim  # 添加其他智能体的动作onehot

        self.cent_obs_dim = cent_obs_dim

        self.act_dim = act_dim
        self._use_orthogonal = args.use_orthogonal

        if multidiscrete_list:
            self.num_q_inps = sum(multidiscrete_list)
        else:
            self.num_q_inps = self.num_agents
        # dimension of the hidden layer of the mixing net
        self.hidden_layer_dim = 32
        # dimension of the hidden layer of each hypernet
        self.hypernet_hidden_dim = args.hypernet_hidden_dim

        if not self.share_hyper_network and self.use_same_share_obs:  # 如果不使用同一个global net且使用同一个global state
            self.hypernet_output_dim = self.num_q_inps * self.hidden_layer_dim
        else:
            self.hypernet_output_dim = self.hidden_layer_dim
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        if not self.use_same_share_obs and not self.share_hyper_network:
            # hypernets output the weight and bias for the 2 layer MLP which takes in the state and agent Qs and outputs Q_tot
            if args.hypernet_layers == 1:
                # each hypernet only has 1 layer to output the weights
                # hyper_w1 outputs weight matrix which is of dimension (hidden_layer_dim x N)
                self.hyper_w1 = nn.Sequential(*[init_(nn.Linear(self.cent_obs_dim, self.hypernet_output_dim))
                                                for _ in range(self.num_q_inps)])
                # hyper_w2 outputs weight matrix which is of dimension (1 x hidden_layer_dim)
                self.hyper_w2 = nn.Sequential(*[init_(nn.Linear(self.cent_obs_dim, self.hypernet_output_dim))
                                                for _ in range(self.num_q_inps)])
            elif args.hypernet_layers == 2:
                # 2 layer hypernets: output dimensions are same as above case
                self.hyper_w1 = nn.Sequential(*[nn.Sequential(
                    init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
                    nn.ReLU(),
                    init_(nn.Linear(self.hypernet_hidden_dim, self.hypernet_output_dim))
                )for _ in range(self.num_q_inps)])
                self.hyper_w2 = nn.Sequential(*[nn.Sequential(
                    init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
                    nn.ReLU(),
                    init_(nn.Linear(self.hypernet_hidden_dim, self.hypernet_output_dim))
                ) for _ in range(self.num_q_inps)])
            # hyper_b1 outputs bias vector of dimension (1 x hidden_layer_dim)
            self.hyper_b1 = nn.Sequential(*[init_(nn.Linear(self.cent_obs_dim, self.hypernet_output_dim))
                                            for _ in range(self.num_q_inps)])
            # hyper_b2 outptus bias vector of dimension (1 x 1)
            self.hyper_b2 = nn.Sequential(*[nn.Sequential(
                init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
                nn.ReLU(),
                init_(nn.Linear(self.hypernet_hidden_dim, 1))
            )for _ in range(self.num_q_inps)])
        else:
            # hypernets output the weight and bias for the 2 layer MLP which takes in the state and agent Qs and outputs Q_tot
            if args.hypernet_layers == 1:
                # each hypernet only has 1 layer to output the weights
                # hyper_w1 outputs weight matrix which is of dimension (hidden_layer_dim x N)
                self.hyper_w1 = init_(nn.Linear(self.cent_obs_dim, self.hypernet_output_dim))
                # hyper_w2 outputs weight matrix which is of dimension (1 x hidden_layer_dim)
                self.hyper_w2 = init_(nn.Linear(self.cent_obs_dim, self.hypernet_output_dim))
            elif args.hypernet_layers == 2:
                # 2 layer hypernets: output dimensions are same as above case
                # if self.mix_local_q:
                #     self.hyper_w1 = nn.Sequential(
                #         init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
                #         nn.ReLU(),
                #         init_(nn.Linear(self.hypernet_hidden_dim, self.hypernet_output_dim * self.num_q_inps))
                #     )
                # else:
                self.hyper_w1 = nn.Sequential(
                    init_(nn.Linear(self.cent_obs_dim_glq, self.hypernet_hidden_dim)),
                    nn.ReLU(),
                    init_(nn.Linear(self.hypernet_hidden_dim, self.hypernet_output_dim))
                )
                self.hyper_w2 = nn.Sequential(
                    init_(nn.Linear(self.cent_obs_dim_glq, self.hypernet_hidden_dim)),
                    nn.ReLU(),
                    init_(nn.Linear(self.hypernet_hidden_dim, self.hypernet_output_dim))
                )
                # hyper_b1 outputs bias vector of dimension (1 x hidden_layer_dim)
            self.hyper_b1 = init_(nn.Linear(self.cent_obs_dim_glq, self.hypernet_output_dim))
            # hyper_b2 outptus bias vector of dimension (1 x 1)
            self.hyper_b2 = nn.Sequential(
                init_(nn.Linear(self.cent_obs_dim_glq, self.hypernet_hidden_dim)),
                nn.ReLU(),
                init_(nn.Linear(self.hypernet_hidden_dim, self.num_q_inps
                if self.use_same_share_obs and not self.share_hyper_network else 1))
            )
        if self.use_global_mixing_network:
            self.hyper_w3 = nn.Sequential(
                init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
                nn.ReLU(),
                init_(nn.Linear(self.hypernet_hidden_dim, self.hypernet_output_dim * 32 * 3))
            )
            self.hyper_b3 = nn.Sequential(
                init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
                nn.ReLU(),
                init_(nn.Linear(self.hypernet_hidden_dim, 32))
            )
            self.hyper_w4 = nn.Sequential(
                init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
                nn.ReLU(),
                init_(nn.Linear(self.hypernet_hidden_dim, 32))
            )
            self.hyper_b4 = nn.Sequential(
                init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
                nn.ReLU(),
                init_(nn.Linear(self.hypernet_hidden_dim, 1))
            )
        self.to(device)

    def forward(self, agent_q_individual, states, actions=None):
        """
         Computes Q_tot using the individual agent q values and global state.
         :param agent_q_inps: (torch.Tensor) individual agent q values
         :param states: (torch.Tensor) state input to the hypernetworks.
         :return Q_tot: (torch.Tensor) computed Q_tot values
         """
        agent_q_individual = to_torch(agent_q_individual).to(**self.tpdv)
        states = to_torch(states).to(**self.tpdv)

        batch_size = agent_q_individual.size(1)
        # if self.add_other_act_to_cent:
        states_glq = torch.cat((states, actions), dim=-1)
        states = states.view(-1, batch_size, self.num_q_inps, self.cent_obs_dim).float()
        states_glq = states_glq.view(-1, batch_size, self.num_q_inps, self.cent_obs_dim_glq).float()
        agent_q_individual = agent_q_individual.view(-1, batch_size, self.num_q_inps, 1, 1)
        # 对hyper网络进行前向传播得到global网络参数
        w1 = torch.abs(self.hyper_w1(states_glq))
        b1 = self.hyper_b1(states_glq)
        w2 = torch.abs(self.hyper_w2(states_glq))
        b2 = self.hyper_b2(states_glq)
        w3 = torch.abs(self.hyper_w3(states[:,:,0,:]))
        b3 = self.hyper_b3(states[:,:,0,:])
        w4 = torch.abs(self.hyper_w4(states[:,:,0,:]))
        b4 = self.hyper_b4(states[:,:,0,:])

        # 参数各个维度分别代表(episode总step数，批量，智能体个数，输入维度（输入一个q），输出维度)
        w1 = w1.view(-1, batch_size, self.num_q_inps, 1, self.hidden_layer_dim)
        b1 = b1.view(-1, batch_size, self.num_q_inps, 1, self.hidden_layer_dim)
        w2 = w2.view(-1, batch_size, self.num_q_inps, self.hidden_layer_dim, 1)
        b2 = b2.view(-1, batch_size, self.num_q_inps, 1, 1)
        w3 = w3.view(-1, batch_size, 1, self.hidden_layer_dim*self.num_q_inps, 32)
        b3 = b3.view(-1, batch_size, 1, 1, 32)
        w4 = w4.view(-1, batch_size, 1, 32, 1)
        b4 = b4.view(-1, batch_size, 1, 1, 1)
        # 对global网络前向传播
        hidden_layer = F.elu(torch.matmul(agent_q_individual, w1) + b1)
        out = torch.matmul(hidden_layer, w2) + b2
        q_global = out.view(-1, batch_size, self.num_q_inps)
        if self.use_global_mixing_network:
            hidden_layer = hidden_layer.view(-1, batch_size, 1, 1, self.hidden_layer_dim*self.num_q_inps)
            hidden_layer = torch.matmul(hidden_layer, w3) + b3
            out2 = torch.matmul(hidden_layer, w4) + b4
            q_tot = out2.view(-1, batch_size)
        return q_global, q_tot if self.use_global_mixing_network else None
