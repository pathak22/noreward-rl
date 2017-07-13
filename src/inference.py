#!/usr/bin/env python
from __future__ import print_function
import go_vncdriver
import tensorflow as tf
import numpy as np
import argparse
import logging
import os
import gym
from envs import create_env
from worker import FastSaver
from model import LSTMPolicy
import utils
import distutils.version
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def inference(args):
    """
    It only restores LSTMPolicy architecture, and does inference using that.
    """
    # get address of checkpoints
    indir = os.path.join(args.log_dir, 'train')
    outdir = os.path.join(args.log_dir, 'inference') if args.out_dir is None else args.out_dir
    with open(indir + '/checkpoint', 'r') as f:
        first_line = f.readline().strip()
    ckpt = first_line.split(' ')[-1].split('/')[-1][:-1]
    ckpt = ckpt.split('-')[-1]
    ckpt = indir + '/model.ckpt-' + ckpt

    # define environment
    if args.record:
        env = create_env(args.env_id, client_id='0', remotes=None, envWrap=args.envWrap, designHead=args.designHead,
                            record=True, noop=args.noop, acRepeat=args.acRepeat, outdir=outdir)
    else:
        env = create_env(args.env_id, client_id='0', remotes=None, envWrap=args.envWrap, designHead=args.designHead,
                            record=True, noop=args.noop, acRepeat=args.acRepeat)
    numaction = env.action_space.n

    with tf.device("/cpu:0"):
        # define policy network
        with tf.variable_scope("global"):
            policy = LSTMPolicy(env.observation_space.shape, numaction, args.designHead)
            policy.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                               trainable=False)

        # Variable names that start with "local" are not saved in checkpoints.
        if use_tf12_api:
            variables_to_restore = [v for v in tf.global_variables() if not v.name.startswith("local")]
            init_all_op = tf.global_variables_initializer()
        else:
            variables_to_restore = [v for v in tf.all_variables() if not v.name.startswith("local")]
            init_all_op = tf.initialize_all_variables()
        saver = FastSaver(variables_to_restore)

        # print trainable variables
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        logger.info('Trainable vars:')
        for v in var_list:
            logger.info('  %s %s', v.name, v.get_shape())

        # summary of rewards
        action_writers = []
        if use_tf12_api:
            summary_writer = tf.summary.FileWriter(outdir)
            for ac_id in range(numaction):
                action_writers.append(tf.summary.FileWriter(os.path.join(outdir,'action_{}'.format(ac_id))))
        else:
            summary_writer = tf.train.SummaryWriter(outdir)
            for ac_id in range(numaction):
                action_writers.append(tf.train.SummaryWriter(os.path.join(outdir,'action_{}'.format(ac_id))))
        logger.info("Inference events directory: %s", outdir)

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=config) as sess:
            logger.info("Initializing all parameters.")
            sess.run(init_all_op)
            logger.info("Restoring trainable global parameters.")
            saver.restore(sess, ckpt)
            logger.info("Restored model was trained for %.2fM global steps", sess.run(policy.global_step)/1000000.)
            # saving with meta graph:
            # metaSaver = tf.train.Saver(variables_to_restore)
            # metaSaver.save(sess, 'models/doomICM')

            last_state = env.reset()
            if args.render or args.record:
                env.render()
            last_features = policy.get_initial_features()  # reset lstm memory
            length = 0
            rewards = 0
            mario_distances = np.zeros((args.num_episodes,))
            for i in range(args.num_episodes):
                print("Starting episode %d" % (i + 1))
                if args.recordSignal:
                    from PIL import Image
                    signalCount = 1
                    utils.mkdir_p(outdir + '/recordedSignal/ep_%02d/'%i)
                    Image.fromarray((255*last_state[..., -1]).astype('uint8')).save(outdir + '/recordedSignal/ep_%02d/%06d.jpg'%(i,signalCount))

                if args.random:
                    print('I am random policy!')
                else:
                    if args.greedy:
                        print('I am greedy policy!')
                    else:
                        print('I am sampled policy!')
                while True:
                    # run policy
                    fetched = policy.act_inference(last_state, *last_features)
                    prob_action, action, value_, features = fetched[0], fetched[1], fetched[2], fetched[3:]

                    # run environment: sampled one-hot 'action' (not greedy)
                    if args.random:
                        stepAct = np.random.randint(0, numaction)  # random policy
                    else:
                        if args.greedy:
                            stepAct = prob_action.argmax()  # greedy policy
                        else:
                            stepAct = action.argmax()
                    # print(stepAct, prob_action.argmax(), prob_action)
                    state, reward, terminal, info = env.step(stepAct)

                    # update stats
                    length += 1
                    rewards += reward
                    last_state = state
                    last_features = features
                    if args.render or args.record:
                        env.render()
                    if args.recordSignal:
                        signalCount += 1
                        Image.fromarray((255*last_state[..., -1]).astype('uint8')).save(outdir + '/recordedSignal/ep_%02d/%06d.jpg'%(i,signalCount))

                    # store summary
                    summary = tf.Summary()
                    summary.value.add(tag='ep_{}/reward'.format(i), simple_value=reward)
                    summary.value.add(tag='ep_{}/netreward'.format(i), simple_value=rewards)
                    summary.value.add(tag='ep_{}/value'.format(i), simple_value=float(value_[0]))
                    if 'NoFrameskip-v' in args.env_id:  # atari
                        summary.value.add(tag='ep_{}/lives'.format(i), simple_value=env.unwrapped.ale.lives())
                    summary_writer.add_summary(summary, length)
                    summary_writer.flush()
                    summary = tf.Summary()
                    for ac_id in range(numaction):
                        summary.value.add(tag='action_prob', simple_value=float(prob_action[ac_id]))
                        action_writers[ac_id].add_summary(summary, length)
                        action_writers[ac_id].flush()

                    timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
                    if timestep_limit is None: timestep_limit = env.spec.timestep_limit
                    if terminal or length >= timestep_limit:
                        if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                            last_state = env.reset()
                        last_features = policy.get_initial_features()  # reset lstm memory
                        print("Episode finished. Sum of rewards: %.2f. Length: %d." % (rewards, length))
                        if 'distance' in info:
                            print('Mario Distance Covered:', info['distance'])
                            mario_distances[i] = info['distance']
                        length = 0
                        rewards = 0
                        if args.render or args.record:
                            env.render()
                        if args.recordSignal:
                            signalCount += 1
                            Image.fromarray((255*last_state[..., -1]).astype('uint8')).save(outdir + '/recordedSignal/ep_%02d/%06d.jpg'%(i,signalCount))
                        break

        logger.info('Finished %d true episodes.', args.num_episodes)
        if 'distance' in info:
            print('Mario Distances:', mario_distances)
            np.save(outdir + '/distances.npy', mario_distances)
        env.close()


def main(_):
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--log-dir', default="tmp/doom", help='input model directory')
    parser.add_argument('--out-dir', default=None, help='output log directory. Default: log_dir/inference/')
    parser.add_argument('--env-id', default="PongDeterministic-v3", help='Environment id')
    parser.add_argument('--record', action='store_true',
                        help="Record the gym environment video -- user friendly")
    parser.add_argument('--recordSignal', action='store_true',
                        help="Record images of true processed input to network")
    parser.add_argument('--render', action='store_true',
                        help="Render the gym environment video online")
    parser.add_argument('--envWrap', action='store_true',
                        help="Preprocess input in env_wrapper (no change in input size or network)")
    parser.add_argument('--designHead', type=str, default='universe',
                        help="Network deign head: nips or nature or doom or universe(default)")
    parser.add_argument('--num-episodes', type=int, default=2,
                        help="Number of episodes to run")
    parser.add_argument('--noop', action='store_true',
                        help="Add 30-noop for inference too (recommended by Nature paper, don't know?)")
    parser.add_argument('--acRepeat', type=int, default=0,
                        help="Actions to be repeated at inference. 0 means default. applies iff envWrap is True.")
    parser.add_argument('--greedy', action='store_true',
                        help="Default sampled policy. This option does argmax.")
    parser.add_argument('--random', action='store_true',
                        help="Default sampled policy. This option does random policy.")
    parser.add_argument('--default', action='store_true', help="run with default params")
    args = parser.parse_args()
    if args.default:
        args.envWrap = True
        args.acRepeat = 1
    if args.acRepeat <= 0:
        print('Using default action repeat (i.e. 4). Min value that can be set is 1.')
    inference(args)

if __name__ == "__main__":
    tf.app.run()
