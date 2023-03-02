import os
import numpy as np
import wandb
import torch
from tensorboardX import SummaryWriter
from torch import multiprocessing as _mp

from offpolicy.utils.rec_buffer import RecReplayBuffer, PrioritizedRecReplayBuffer
from offpolicy.utils.util import DecayThenFlatSchedule
from offpolicy.runner.rnn.make_env import make_train_env, make_eval_env  # 用于多进程新建环境


class RecRunner(object):
    """Base class for training recurrent policies."""

    def __init__(self, config):
        """
        Base class for training recurrent policies.
        :param config: (dict) Config dictionary containing parameters for training.
        """
        self.args = config["args"]
        self.device = config["device"]
        self.q_learning = ["qmix","vdn"]
        self.global_and_local_q = ["glq", "glq_gr"]
        self.run_dir = config["run_dir"]

        self.share_policy = self.args.share_policy
        self.algorithm_name = self.args.algorithm_name
        self.env_name = self.args.env_name
        self.num_env_steps = self.args.num_env_steps
        self.use_wandb = self.args.use_wandb
        self.use_reward_normalization = self.args.use_reward_normalization
        self.use_popart = self.args.use_popart
        self.use_per = self.args.use_per
        self.per_alpha = self.args.per_alpha
        self.per_beta_start = self.args.per_beta_start
        self.buffer_size = self.args.buffer_size
        self.batch_size = self.args.batch_size
        self.hidden_size = self.args.hidden_size
        self.use_soft_update = self.args.use_soft_update
        self.hard_update_interval_episode = self.args.hard_update_interval_episode
        self.popart_update_interval_step = self.args.popart_update_interval_step
        self.actor_train_interval_step = self.args.actor_train_interval_step
        self.train_interval_episode = self.args.train_interval_episode
        self.train_interval = self.args.train_interval
        self.use_asynchronous_eval = self.args.use_asynchronous_eval
        self.eval_interval = self.args.eval_interval
        self.save_interval = self.args.save_interval
        self.log_interval = self.args.log_interval
        self.save_all = self.args.save_all

        self.total_env_steps = 0  # total environment interactions collected during training
        self.num_episodes_collected = 0  # total episodes collected during training
        self.total_train_steps = 0  # number of gradient updates performed
        self.last_train_episode = 0  # last episode after which a gradient update was performed
        self.last_eval_T = 0  # last episode after which a eval run was conducted
        self.last_save_T = 0  # last epsiode after which the models were saved
        self.last_log_T = 0 # last timestep after which information was logged
        self.last_hard_update_episode = 0 # last episode after which target policy was updated to equal live policy

        if config.__contains__("take_turn"):
            self.take_turn = config["take_turn"]
        else:
            self.take_turn = False

        if config.__contains__("if_train"):
            self.if_train = config["if_train"]
        else:
            self.if_train = True

        if config.__contains__("render_interval"):
            self.render_interval = config["render_interval"]
        else:
            self.render_interval = 0

        if config.__contains__("use_same_share_obs"):
            self.use_same_share_obs = config["use_same_share_obs"]
        else:
            self.use_same_share_obs = False

        if config.__contains__("use_available_actions"):
            self.use_avail_acts = config["use_available_actions"]
        else:
            self.use_avail_acts = False

        if config.__contains__("buffer_length"):
            self.episode_length = config["buffer_length"]
            if self.args.use_naive_recurrent_policy:
                self.data_chunk_length = config["buffer_length"]
            else:
                self.data_chunk_length = self.args.data_chunk_length
        else:
            self.episode_length = self.args.episode_length
            if self.args.use_naive_recurrent_policy:
                self.data_chunk_length = self.args.episode_length
            else:
                self.data_chunk_length = self.args.data_chunk_length

        self.policy_info = config["policy_info"]

        self.policy_ids = sorted(list(self.policy_info.keys()))
        self.num_agents = config["num_agents"]
        self.agent_ids = [i for i in range(self.num_agents)]

        try:
            self.new_proc_eval = config['new_proc_eval']
        except BaseException as e:
            self.new_proc_eval = None
        # no parallel envs
        self.num_envs = config['env'].num_envs
        # self.mp = _mp.get_context("spawn")
        # self.start_mp_evaluation(config['args'].new_proc_eval_render)

        # initialize all the policies and organize the agents corresponding to each policy
        if self.algorithm_name == "glq":
            from offpolicy.algorithms.glq.algorithm.GLQPolicy import GLQPolicy as Policy
            from offpolicy.algorithms.glq.glq import GLQ as TrainAlgo
        elif self.algorithm_name == "glq_addQmix":
            from offpolicy.algorithms.glq_addQmix.algorithm.GLQPolicy import GLQPolicy as Policy
            from offpolicy.algorithms.glq_addQmix.glq import GLQ as TrainAlgo
        elif self.algorithm_name == "glq_mixGQ":
            from offpolicy.algorithms.glq_mixGQ.algorithm.GLQPolicy import GLQPolicy as Policy
            from offpolicy.algorithms.glq_mixGQ.glq import GLQ as TrainAlgo
        elif self.algorithm_name == "glq_mixLQ":
            from offpolicy.algorithms.glq_mixLQ.algorithm.GLQPolicy import GLQPolicy as Policy
            from offpolicy.algorithms.glq_mixLQ.glq import GLQ as TrainAlgo
        elif self.algorithm_name == "glq_Qbias":
            from offpolicy.algorithms.glq_Qbias.algorithm.GLQPolicy import GLQPolicy as Policy
            from offpolicy.algorithms.glq_Qbias.glq import GLQ as TrainAlgo
        elif self.algorithm_name == "glq_mixLayer":
            from offpolicy.algorithms.glq_mixLayer.algorithm.GLQPolicy import GLQPolicy as Policy
            from offpolicy.algorithms.glq_mixLayer.glq import GLQ as TrainAlgo
        elif self.algorithm_name == "rmatd3":
            from offpolicy.algorithms.r_matd3.algorithm.rMATD3Policy import R_MATD3Policy as Policy
            from offpolicy.algorithms.r_matd3.r_matd3 import R_MATD3 as TrainAlgo
        elif self.algorithm_name == "rmaddpg":
            assert self.actor_train_interval_step == 1, (
                "rmaddpg only supports actor_train_interval_step=1.")
            from offpolicy.algorithms.r_maddpg.algorithm.rMADDPGPolicy import R_MADDPGPolicy as Policy
            from offpolicy.algorithms.r_maddpg.r_maddpg import R_MADDPG as TrainAlgo
        elif self.algorithm_name == "rmasac":
            assert self.actor_train_interval_step == 1, (
                "rmasac only support actor_train_interval_step=1.")
            from offpolicy.algorithms.r_masac.algorithm.rMASACPolicy import R_MASACPolicy as Policy
            from offpolicy.algorithms.r_masac.r_masac import R_MASAC as TrainAlgo
        elif self.algorithm_name == "qmix":
            from offpolicy.algorithms.qmix.algorithm.QMixPolicy import QMixPolicy as Policy
            from offpolicy.algorithms.qmix.qmix import QMix as TrainAlgo
        elif self.algorithm_name == "vdn":
            from offpolicy.algorithms.vdn.algorithm.VDNPolicy import VDNPolicy as Policy
            from offpolicy.algorithms.vdn.vdn import VDN as TrainAlgo
        else:
            raise NotImplementedError
        
        self.collecter = self.collect_rollout
        if self.algorithm_name in self.q_learning:
            self.saver = self.save_q
            self.train = self.batch_train_q
            self.restorer = self.restore_q
        elif self.algorithm_name[:3] == 'glq':
            self.saver = self.save_glq
            self.train = self.batch_train_glq
            self.restorer = self.restore_glq
        else:
            self.saver = self.save
            self.train = self.batch_train
            self.restorer = self.restore

        self.policies = {p_id: Policy(config, self.policy_info[p_id]) for p_id in self.policy_ids}
        self.policies_eval_mp = {p_id: Policy(config, self.policy_info[p_id]) for p_id in self.policy_ids}


        # initialize trainer class for updating policies
        self.trainer = TrainAlgo(self.args, self.num_agents, self.policies, self.policy_mapping_fn,
                                 device=self.device, episode_length=self.episode_length)


        # map policy id to agent ids controlled by that policy
        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in self.agent_ids if self.policy_mapping_fn(agent_id) == policy_id]) for policy_id in
            self.policies.keys()}

        self.policy_obs_dim = {
            policy_id: self.policies[policy_id].obs_dim for policy_id in self.policy_ids}
        self.policy_act_dim = {
            policy_id: self.policies[policy_id].act_dim for policy_id in self.policy_ids}
        self.policy_central_obs_dim = {
            policy_id: self.policies[policy_id].central_obs_dim for policy_id in self.policy_ids}

        num_train_episodes = (self.num_env_steps / self.episode_length) / (self.train_interval_episode)
        self.beta_anneal = DecayThenFlatSchedule(
            self.per_beta_start, 1.0, num_train_episodes, decay="linear")

        if self.use_per:
            self.buffer = PrioritizedRecReplayBuffer(self.per_alpha,
                                                     self.policy_info,
                                                     self.policy_agents,
                                                     self.buffer_size,
                                                     self.episode_length,
                                                     self.use_same_share_obs,
                                                     self.use_avail_acts,
                                                     self.use_reward_normalization)
        else:
            self.buffer = RecReplayBuffer(self.policy_info,
                                          self.policy_agents,
                                          self.buffer_size,
                                          self.episode_length,
                                          self.use_same_share_obs,
                                          self.use_avail_acts,
                                          self.use_reward_normalization)
        # 待定义
        self.collecter = self.collect_rollout

        # 开新进程评估
        if self.new_proc_eval:
            self.mp = _mp.get_context("spawn")

            self.start_eval = self.mp.Value('b', False)
            self.stop_eval = self.mp.Value('b', False)
            if self.algorithm_name in self.global_and_local_q or self.algorithm_name in self.q_learning:
                for p_id in self.policy_ids:
                    self.policies[p_id].q_network.share_memory()
            self.start_mp_evaluation(config['args'].new_proc_eval_render)

        self.env = config["env"]  # 放在开新进程后面防止报错
        if self.use_asynchronous_eval:
            self.eval_env = config["eval_env"]
        self.config = config
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # Load Models
        self.model_dir = self.args.model_dir
        if self.model_dir is not None and not config['if_train']:
            self.restorer()

    def start_mp_evaluation(self, new_proc_eval_render):
        p = self.mp.Process(target=self.new_processing_eval, args=(new_proc_eval_render,))
        p.start()
        # p.join()

    def new_processing_eval(self, render=False):
        self.env = make_train_env(self.args)
        self.eval_env = make_eval_env(self.args)
        self.model_dir = self.args.model_dir
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:

            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        while not self.start_eval.value:
            if self.stop_eval.value:
                break
        while not self.stop_eval.value:
            self.eval(render)

    def policy_mapping_fn(self, agent_id):
        if self.share_policy:
            return 'policy_0'
        else:
            return 'policy_' + str(agent_id)

    def run(self):
        """Collect a training episode and perform appropriate training, saving, logging, and evaluation steps."""
        # train
        if self.if_train:
            # collect data
            self.trainer.prep_rollout()
            env_info = self.collecter(explore=True, training_episode=True, warmup=False, render=self.args.train_render)
            for k, v in env_info.items():
                self.env_infos[k].append(v)
            if ((self.num_episodes_collected - self.last_train_episode) / self.train_interval_episode) >= 1 or self.last_train_episode == 0:
                self.train()
                self.total_train_steps += 1
                self.last_train_episode = self.num_episodes_collected

            # save
            if (self.total_env_steps - self.last_save_T) / self.save_interval >= 1:
                self.saver()
                self.last_save_T = self.total_env_steps

            # log
            if ((self.total_env_steps - self.last_log_T) / self.log_interval) >= 1:
                self.log()
                self.last_log_T = self.total_env_steps

            # eval
            if self.use_asynchronous_eval and ((self.total_env_steps - self.last_eval_T) / self.eval_interval) >= 1:
                self.eval()
                self.last_eval_T = self.total_env_steps

        else:
            self.eval()

        return self.total_env_steps
    
    def warmup(self, num_warmup_episodes):
        """
        Fill replay buffer with enough episodes to begin training.

        :param: num_warmup_episodes (int): number of warmup episodes to collect.
        """
        self.trainer.prep_rollout()
        warmup_rewards = []
        print("warm up...")
        for _ in range((num_warmup_episodes // self.num_envs) + 1):
            env_info = self.collecter(explore=True, training_episode=False, warmup=True)
            warmup_rewards.append(env_info['episode_rewards'])
        warmup_reward = np.mean(warmup_rewards)
        print("warmup average episode rewards: {}".format(warmup_reward))

    def batch_train(self):
        """Do a gradient update for all policies."""
        self.trainer.prep_training()

        # gradient updates
        self.train_infos = []
        update_actor = False
        for p_id in self.policy_ids:
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)

            update_method = self.trainer.shared_train_policy_on_batch if self.use_same_share_obs else self.trainer.cent_train_policy_on_batch
            
            train_info, new_priorities, idxes = update_method(p_id, sample)
            update_actor = train_info['update_actor']

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)

        if self.use_soft_update and update_actor:
            for pid in self.policy_ids:
                self.policies[pid].soft_target_updates()
        else:
            if ((self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode) >= 1:
                for pid in self.policy_ids:
                    self.policies[pid].hard_target_updates()
                self.last_hard_update_episode = self.num_episodes_collected

    def batch_train_q(self):
        """Do a q-learning update to policy (used for QMix and VDN)."""
        self.trainer.prep_training()
        # gradient updates
        self.train_infos = []

        for p_id in self.policy_ids:
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)

            train_info, new_priorities, idxes = self.trainer.train_policy_on_batch(sample)

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)

        if self.use_soft_update:
            self.trainer.soft_target_updates()
        else:
            if (self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode >= 1:
                self.trainer.hard_target_updates()
                self.last_hard_update_episode = self.num_episodes_collected

    def batch_train_glq(self):
        """Do a q-learning update to policy (Used for GLQ-base and GLQ-gr)."""
        self.trainer.prep_training()
        # gradient updates
        self.train_infos = []
        for p_id in self.policy_ids:
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)

            train_info, new_priorities, idxes = self.trainer.train_policy_on_batch(sample, self.use_same_share_obs,
                                                                                   self.num_agents)

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)

        if self.use_soft_update:
            self.trainer.soft_target_updates()
        else:
            if (self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode >= 1:
                self.trainer.hard_target_updates()
                self.last_hard_update_episode = self.num_episodes_collected

    def save(self):
        """Save all policies to the path specified by the config."""
        for pid in self.policy_ids:
            policy_critic = self.policies[pid].critic
            critic_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(critic_save_path):
                os.makedirs(critic_save_path)
            torch.save(policy_critic.state_dict(),
                       critic_save_path + '/critic.pt')

            policy_actor = self.policies[pid].actor
            actor_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(actor_save_path):
                os.makedirs(actor_save_path)
            torch.save(policy_actor.state_dict(),
                       actor_save_path + '/actor.pt')

    def save_q(self):
        """Save all policies to the path specified by the config. Used for QMix and VDN."""
        for pid in self.policy_ids:
            policy_Q = self.policies[pid].q_network
            p_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(p_save_path):
                os.makedirs(p_save_path)
            torch.save(policy_Q.state_dict(), p_save_path + '/q_network.pt')
            if self.save_all:
                torch.save(policy_Q.state_dict(), p_save_path + '/' + str(self.total_env_steps) + '.q_network.pt')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self.trainer.mixer.state_dict(),
                   self.save_dir + '/mixer.pt')

    def save_glq(self):
        """Save all policies to the path specified by the config, Used for GLQ-base and GLQ-gr."""
        for pid in self.policy_ids:
            policy_Q = self.policies[pid].q_network
            p_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(p_save_path):
                os.makedirs(p_save_path)
            torch.save(policy_Q.state_dict(), p_save_path + '/q_network.pt')
            if self.save_all:
                torch.save(policy_Q.state_dict(), p_save_path + '/' + str(self.total_env_steps) + '.q_network.pt')
        global_q = self.trainer.global_q
        gq_save_path = self.save_dir
        if not os.path.exists(gq_save_path):
            os.makedirs(gq_save_path)
        torch.save(global_q.state_dict(),
                   gq_save_path + '/global_q.pt')

    def restore(self):
        """Load policies policies from pretrained models specified by path in config."""
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print("load the pretrained model from {}".format(path))
            policy_critic_state_dict = torch.load(path + '/critic.pt')
            policy_actor_state_dict = torch.load(path + '/actor.pt')

            self.policies[pid].critic.load_state_dict(policy_critic_state_dict)
            self.policies[pid].actor.load_state_dict(policy_actor_state_dict)

    def restore_q(self):
        """Load policies policies from pretrained models specified by path in config. Used for QMix and VDN."""
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print("load the pretrained model from {}".format(path))
            policy_q_state_dict = torch.load(path + '/q_network.pt')           
            self.policies[pid].q_network.load_state_dict(policy_q_state_dict)
        if self.if_train:
            policy_mixer_state_dict = torch.load(str(self.model_dir) + '/mixer.pt')
            self.trainer.mixer.load_state_dict(policy_mixer_state_dict)

    def restore_glq(self):
        """Load policies policies from pretrained models specified by path in config. Used for GLQ-base and GLQ-gr."""
        if self.if_train:
            print("load the pretrained global q model from {}".format(self.model_dir))
            global_q_state_dict = torch.load(self.model_dir + '/global_q.pt')
            self.trainer.global_q.load_state_dict(global_q_state_dict)
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print("load the pretrained local q model from {}".format(path))
            policy_q_state_dict = torch.load(path + '/q_network.pt')
            self.policies[pid].q_network.load_state_dict(policy_q_state_dict)

    def log(self):
        """Log relevent training and rollout colleciton information.."""
        raise NotImplementedError

    def log_clear(self):
        """Clear logging variables so they do not contain stale information."""
        raise NotImplementedError

    def log_env(self, env_info, suffix=None):
        """
        Log information related to the environment.
        :param env_info: (dict) contains logging information related to the environment.
        :param suffix: (str) optional string to add to end of keys in env_info when logging. 
        """
        for k, v in env_info.items():
            if len(v) > 0:
                v = np.mean(v)
                suffix_k = k if suffix is None else suffix + k 
                print(suffix_k + " is " + str(v))
                if self.use_wandb:
                    wandb.log({suffix_k: v}, step=self.total_env_steps)
                else:
                    self.writter.add_scalars(suffix_k, {suffix_k: v}, self.total_env_steps)

    def log_train(self, policy_id, train_info):
        """
        Log information related to training.
        :param policy_id: (str) policy id corresponding to the information contained in train_info.
        :param train_info: (dict) contains logging information related to training.
        """
        for k, v in train_info.items():
            policy_k = str(policy_id) + '/' + k
            if self.use_wandb:
                wandb.log({policy_k: v}, step=self.total_env_steps)
            else:
                self.writter.add_scalars(policy_k, {policy_k: v}, self.total_env_steps)

    def collect_rollout(self):
        """Collect a rollout and store it in the buffer."""
        raise NotImplementedError