import torch
import copy
from offpolicy.utils.util import soft_update, huber_loss, mse_loss, to_torch
from offpolicy.algorithms.glq_mixGQ.algorithm.q_global import q_global
from offpolicy.algorithms.base.trainer import Trainer
from offpolicy.utils.popart import PopArt
import numpy as np

class GLQ(Trainer):
    def __init__(self, args, num_agents, policies, policy_mapping_fn, device=torch.device("cuda:0"), episode_length=None, gr=False):
        """
        Trainer class for recurrent QMix/VDN. See parent class for more information.
        :param episode_length: (int) maximum length of an episode.
        :param vdnl: (bool) whether the algorithm being used is VDN.
        """
        self.args = args

        self.add_other_act_to_cent = args.add_other_act_to_cent

        self.use_popart = self.args.use_popart
        self.use_value_active_masks = self.args.use_value_active_masks
        self.use_per = self.args.use_per
        self.per_eps = self.args.per_eps
        self.use_huber_loss = self.args.use_huber_loss
        self.huber_delta = self.args.huber_delta

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.lr = self.args.lr
        self.tau = self.args.tau
        self.opti_eps = self.args.opti_eps
        self.weight_decay = self.args.weight_decay

        if episode_length is None:
            self.episode_length = self.args.episode_length
        else:
            self.episode_length = episode_length

        self.num_agents = num_agents
        self.policies = policies
        self.policy_mapping_fn = policy_mapping_fn
        self.policy_ids = sorted(list(self.policies.keys()))
        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in range(self.num_agents) if self.policy_mapping_fn(agent_id) == policy_id])
            for policy_id in self.policies.keys()}
        if self.use_popart:
            self.value_normalizer = {policy_id: PopArt(1) for policy_id in self.policies.keys()}

        self.use_same_share_obs = self.args.use_same_share_obs

        multidiscrete_list = None
        if any([isinstance(policy.act_dim, np.ndarray) for policy in self.policies.values()]):
            # multidiscrete
            multidiscrete_list = [len(self.policies[p_id].act_dim) *
                                  len(self.policy_agents[p_id]) for p_id in self.policy_ids]

        # global network
        if not gr:
            self.global_q = q_global(args, self.num_agents, self.policies['policy_0'].central_obs_dim,
                                     self.policies['policy_0'].act_dim, self.device)

        # target policies/networks
        self.target_policies = {p_id: copy.deepcopy(self.policies[p_id]) for p_id in self.policy_ids}
        self.target_global_q = copy.deepcopy(self.global_q)

        # collect all trainable parameters: each policy parameters, and the mixer parameters
        self.parameters = []
        for policy in self.policies.values():
            self.parameters += policy.parameters()
        self.parameters += self.global_q.parameters()
        self.optimizer = torch.optim.Adam(
            params=self.parameters, lr=self.lr, eps=self.opti_eps)

        if self.args.use_double_q:
            print("double Q learning will be used")


    def train_policy_on_batch(self, batch, update_policy_id=None, num_agents=None, env_step=0):
        """See parent class."""
        # unpack the batch
        obs_batch, cent_obs_batch, \
        act_batch, rew_batch, \
        dones_batch, dones_env_batch, \
        avail_act_batch, \
        importance_weights, idxes = batch
        num_q_inps = num_agents
        cent_obs_batch = to_torch(cent_obs_batch[self.policy_ids[0]])

        dones_env_batch = to_torch(dones_env_batch[self.policy_ids[0]]).to(**self.tpdv).unsqueeze(-1)

        losses = []
        grads_norm = []
        Qs_global = []
        Qs_tot = []
        for p_id in self.policy_ids:
            policy = self.policies[p_id]
            target_policy = self.target_policies[p_id]
            # get data related to the policy id
            pol_obs_batch = to_torch(obs_batch[p_id])
            curr_act_batch = to_torch(act_batch[p_id]).to(**self.tpdv)

            # stack over policy's agents to process them at once
            stacked_act_batch = torch.cat(list(curr_act_batch), dim=-2)
            stacked_obs_batch = torch.cat(list(pol_obs_batch), dim=-2)

            if avail_act_batch[p_id] is not None:
                curr_avail_act_batch = to_torch(avail_act_batch[p_id])
                stacked_avail_act_batch = torch.cat(list(curr_avail_act_batch), dim=-2)
            else:
                stacked_avail_act_batch = None

            # [num_agents, episode_length, episodes, dim]
            batch_size = pol_obs_batch.shape[2]
            total_batch_size = batch_size * len(self.policy_agents[p_id])
            sum_act_dim = int(sum(policy.act_dim)) if policy.multidiscrete else policy.act_dim
            # 排除掉自己的动作，得到其他智能体动作，作为全局信息输入q global
            if self.add_other_act_to_cent:
                curr_other_act_batch = torch.cat([torch.cat(list(curr_act_batch[list(range(0, n)) + list(range(n+1, num_agents))]), dim=-1).unsqueeze(dim=-2)
                                                  for n in range(num_agents)], dim=-2)
                next_other_act_batch = torch.cat((curr_other_act_batch[1:],
                                                     torch.zeros(1, batch_size, num_agents, sum_act_dim*(num_agents-1)).to(**self.tpdv)))

            pol_prev_act_buffer_seq = torch.cat((torch.zeros(1, total_batch_size, sum_act_dim).to(**self.tpdv),
                                                 stacked_act_batch))

            # sequence of q values for all possible actions
            pol_all_q_seq, _ = policy.get_q_values(stacked_obs_batch, pol_prev_act_buffer_seq,
                                                                            policy.init_hidden(-1, total_batch_size))
            # get only the q values corresponding to actions taken in action_batch.
            # '''Ignore the last time dimension.'''
            if policy.multidiscrete:
                pol_all_q_curr_seq = [q_seq[:-1] for q_seq in pol_all_q_seq]
                pol_q_seq = policy.q_values_from_actions(pol_all_q_curr_seq, stacked_act_batch)
            else:
                pol_q_seq = policy.q_values_from_actions(pol_all_q_seq[:-1], stacked_act_batch)
            agent_q_out_sequence = pol_q_seq.split(split_size=batch_size, dim=-2)
            agent_q_seq = torch.cat(agent_q_out_sequence, dim=-1)

            with torch.no_grad():
                if self.args.use_double_q:
                    # choose greedy actions from live, but get corresponding q values from target
                    greedy_actions, _ = policy.actions_from_q(pol_all_q_seq, available_actions=stacked_avail_act_batch)
                    target_q_seq, _ = target_policy.get_q_values(stacked_obs_batch, pol_prev_act_buffer_seq, target_policy.init_hidden(-1, total_batch_size), action_batch=greedy_actions)
                else:
                    _, _, target_q_seq = target_policy.get_actions(stacked_obs_batch, pol_prev_act_buffer_seq, target_policy.init_hidden(-1, total_batch_size))
            # don't need the first Q values for next step
            target_q_seq = target_q_seq[1:]
            agent_nq_sequence = target_q_seq.split(split_size=batch_size, dim=-2)
            agent_nq_seq = torch.cat(agent_nq_sequence, dim=-1)

            if not self.use_same_share_obs:
                stacked_cent_obs_batch = torch.cat(list(cent_obs_batch.unsqueeze(-2)), dim=-2)
            else:
                stacked_cent_obs_batch = cent_obs_batch
            # get curr step and next step Q_tot values using mixer
            if self.add_other_act_to_cent:
                Q_global_seq, Q_tot_seq = self.global_q(agent_q_seq, stacked_cent_obs_batch[:-1], curr_other_act_batch)
                next_step_Q_global_seq, next_step_Q_tot_seq = self.target_global_q(agent_nq_seq, stacked_cent_obs_batch[1:], next_other_act_batch)
            else:
                Q_global_seq, Q_tot_seq = self.global_q(agent_q_seq, stacked_cent_obs_batch[:-1])
                next_step_Q_global_seq, next_step_Q_tot_seq = self.target_global_q(agent_nq_seq, stacked_cent_obs_batch[1:])
            Q_global_seq = Q_global_seq.unsqueeze(-1)
            # Q_tot_seq = Q_tot_seq.unsqueeze(-1).unsqueeze(-1)
            next_step_Q_global_seq = next_step_Q_global_seq.unsqueeze(-1)
            # next_step_Q_tot_seq = next_step_Q_tot_seq.unsqueeze(-1).unsqueeze(-1)
            rewards = torch.cat([to_torch(rew_batch[p_id][n])for n in range(num_q_inps)],dim=-1).to(**self.tpdv)
            # global_rewards = rewards.mean(-1)
            rewards = rewards.unsqueeze(-1)
            # global_rewards = global_rewards.unsqueeze(-1).unsqueeze(-1)
            # form bad transition mask
            bad_transitions_mask = torch.cat((torch.zeros(1, batch_size, 1, 1).to(**self.tpdv), dones_env_batch[:self.episode_length - 1, :, :, :]))

            # bootstrapped targets
            Q_global_target_seq = rewards + (1 - dones_env_batch) * self.args.gamma * next_step_Q_global_seq
            # Q_tot_target_seq = global_rewards + (1 - dones_env_batch) * self.args.gamma * next_step_Q_tot_seq
            # Bellman error and mask out invalid transitions
            # eps = self.policies[p_id].exploration.eval(env_step)
            # print("glq占比：", eps)
            error1 = (Q_global_seq - Q_global_target_seq.detach()) * (1 - bad_transitions_mask)
            # error2 = (Q_tot_seq - Q_tot_target_seq.detach()) * (1 - bad_transitions_mask)

            if self.use_huber_loss:
                loss = huber_loss(error1, self.huber_delta).sum() / (1 - bad_transitions_mask).sum()
            else:
                loss = mse_loss(error1).sum() / (1 - bad_transitions_mask).sum()
            # if self.use_huber_loss:
            #     loss2 = huber_loss(error2, self.huber_delta).sum() / (1 - bad_transitions_mask).sum()
            # else:
            #     loss2 = mse_loss(error2).sum() / (1 - bad_transitions_mask).sum()
            # loss = eps*loss1 + (1-eps)*loss2
            # backward pass and gradient step
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.args.max_grad_norm)
            self.optimizer.step()
            losses.append(loss)
            grads_norm.append(grad_norm)
            Qs_global.append((Q_global_seq * (1 - bad_transitions_mask)).mean())
        new_priorities = None
        # log
        train_info = {}
        train_info['loss'] = sum(losses)/len(self.policy_ids)
        train_info['grad_norm'] = sum(grads_norm)/len(self.policy_ids)
        train_info['Q_global'] = sum(Qs_global)/len(self.policy_ids)
        train_info['Q_tot'] = sum(Qs_tot)/len(self.policy_ids)
        return train_info, new_priorities, idxes


    def hard_target_updates(self):
        """Hard update the target networks."""
        print("hard update targets")
        for policy_id in self.policy_ids:
            self.target_policies[policy_id].load_state(self.policies[policy_id])
        if self.global_q is not None:
            self.target_global_q.load_state_dict(self.global_q.state_dict())

    def soft_target_updates(self):
        """Soft update the target networks."""
        for policy_id in self.policy_ids:
            soft_update(self.target_policies[policy_id], self.policies[policy_id], self.tau)
        if self.global_q is not None:
            soft_update(self.target_global_q, self.global_q, self.tau)

    def prep_training(self):
        """See parent class."""
        for p_id in self.policy_ids:
            self.policies[p_id].q_network.train()
            self.target_policies[p_id].q_network.train()
        self.global_q.train()
        self.target_global_q.train()

    def prep_rollout(self):
        """See parent class."""
        for p_id in self.policy_ids:
            self.policies[p_id].q_network.eval()
            self.target_policies[p_id].q_network.eval()
        self.global_q.eval()
        self.target_global_q.eval()
