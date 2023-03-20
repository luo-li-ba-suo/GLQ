import sys
import os
import numpy as np
from pathlib import Path
import socket
import wandb
import setproctitle
import torch
from offpolicy.config import get_config
from offpolicy.utils.util import get_cent_act_dim, get_dim_from_space
from offpolicy.envs.mpe.MPE_Env import MPEEnv
from offpolicy.envs.env_wrappers import DummyVecEnv, SubprocVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=3, help="number of agents")

    # For JointTag
    parser.add_argument('--num_good_agents', type=int,
                        default=2, help="number of agents")
    parser.add_argument('--num_adversaries', type=int,
                        default=4, help="number of agents")
    parser.add_argument('--num_joint', type=int,
                        default=2, help="number of joint tag agents")
    parser.add_argument('--num_protect_frame', type=int,
                        default=5, help="number of frames in which the good agents get rid of adversaries")
    all_args = parser.parse_known_args(args)[0]

    if all_args.scenario_name == "joint_tag":
        all_args.num_agents = all_args.num_adversaries
    if all_args.algorithm_name == "qmix":
        all_args.num_perspective = 1
    if all_args.algorithm_name == "mpqmix":
        all_args.use_same_share_obs = False
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda and # threads
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # setup file to output tensorboard, hyperparameters, and saved models
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        # init wandb
        run = wandb.init(config=all_args,
                         project="GLQ_Beta",
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[
                                     1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(
        all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # set seeds
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # create env
    env = make_train_env(all_args)
    num_agents = all_args.num_agents

    # create policies and mapping fn
    if all_args.share_policy:
        policy_info = {
            'policy_0': {"cent_obs_dim": get_dim_from_space(env.share_observation_space[0]),
                         "cent_act_dim": get_cent_act_dim(env.action_space),
                         "obs_space": env.observation_space[0],
                         "share_obs_space": env.share_observation_space[0],
                         "act_space": env.action_space[0]}
        }

    else:
        policy_info = {
            'policy_' + str(agent_id): {"cent_obs_dim": get_dim_from_space(env.share_observation_space[agent_id]),
                                        "cent_act_dim": get_cent_act_dim(env.action_space),
                                        "obs_space": env.observation_space[agent_id],
                                        "share_obs_space": env.share_observation_space[agent_id],
                                        "act_space": env.action_space[agent_id]}
            for agent_id in range(num_agents)
        }

    # choose algo
    if all_args.algorithm_name in ["rmatd3", "rmaddpg", "rmasac", "qmix", "vdn", "glq", "glq_addQmix",
                                   "glq_mixGQ", "glq_mixLQ", "glq_Qbias", "glq_mixLayer", "mpqmix"]:
        from offpolicy.runner.rnn.mpe_runner import MPERunner as Runner
    elif all_args.algorithm_name in ["matd3", "maddpg", "masac", "mqmix", "mvdn"]:
        from offpolicy.runner.mlp.mpe_runner import MPERunner as Runner
    else:
        raise NotImplementedError
    eval_env = make_eval_env(all_args)
    if all_args.new_proc_eval:
        all_args.use_asynchronous_eval = False
    config = {"args": all_args,
              "policy_info": policy_info,
              "env": env,
              "eval_env": eval_env,
              "num_agents": num_agents,
              "device": device,
              "use_same_share_obs": all_args.use_same_share_obs,
              "run_dir": run_dir,
              "new_proc_eval": all_args.new_proc_eval,
              "if_train": all_args.if_train,
              "render_interval": all_args.render_interval
              }

    total_num_steps = 0
    runner = Runner(config=config)

    while total_num_steps < all_args.num_env_steps:
        total_num_steps = runner.run()

    env.close()
    if all_args.use_asynchronous_eval and (eval_env is not env):
        eval_env.close()
    if all_args.new_proc_eval:
        Runner.stop_eval.value = True

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(
            str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
