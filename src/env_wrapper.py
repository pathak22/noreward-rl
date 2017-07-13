"""
Author: Deepak Pathak

Acknowledgement:
    - The wrappers (BufferedObsEnv, SkipEnv) were originally written by
        Evan Shelhamer and modified by Deepak. Thanks Evan!
    - This file is derived from
        https://github.com/shelhamer/ourl/envs.py
        https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers_deprecated.py
"""
from __future__ import print_function
import numpy as np
from collections import deque
from PIL import Image
from gym.spaces.box import Box
import gym
import time, sys


class BufferedObsEnv(gym.ObservationWrapper):
    """Buffer observations and stack e.g. for frame skipping.

    n is the length of the buffer, and number of observations stacked.
    skip is the number of steps between buffered observations (min=1).

    n.b. first obs is the oldest, last obs is the newest.
         the buffer is zeroed out on reset.
         *must* call reset() for init!
    """
    def __init__(self, env=None, n=4, skip=4, shape=(84, 84),
                    channel_last=True, maxFrames=True):
        super(BufferedObsEnv, self).__init__(env)
        self.obs_shape = shape
        # most recent raw observations (for max pooling across time steps)
        self.obs_buffer = deque(maxlen=2)
        self.maxFrames = maxFrames
        self.n = n
        self.skip = skip
        self.buffer = deque(maxlen=self.n)
        self.counter = 0  # init and reset should agree on this
        shape = shape + (n,) if channel_last else (n,) + shape
        self.observation_space = Box(0.0, 255.0, shape)
        self.ch_axis = -1 if channel_last else 0
        self.scale = 1.0 / 255
        self.observation_space.high[...] = 1.0

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info

    def _observation(self, obs):
        obs = self._convert(obs)
        self.counter += 1
        if self.counter % self.skip == 0:
            self.buffer.append(obs)
        obsNew = np.stack(self.buffer, axis=self.ch_axis)
        return obsNew.astype(np.float32) * self.scale

    def _reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        self.obs_buffer.clear()
        obs = self._convert(self.env.reset())
        self.buffer.clear()
        self.counter = 0
        for _ in range(self.n - 1):
            self.buffer.append(np.zeros_like(obs))
        self.buffer.append(obs)
        obsNew = np.stack(self.buffer, axis=self.ch_axis)
        return obsNew.astype(np.float32) * self.scale

    def _convert(self, obs):
        self.obs_buffer.append(obs)
        if self.maxFrames:
            max_frame = np.max(np.stack(self.obs_buffer), axis=0)
        else:
            max_frame = obs
        intensity_frame = self._rgb2y(max_frame).astype(np.uint8)
        small_frame = np.array(Image.fromarray(intensity_frame).resize(
            self.obs_shape, resample=Image.BILINEAR), dtype=np.uint8)
        return small_frame

    def _rgb2y(self, im):
        """Converts an RGB image to a Y image (as in YUV).

        These coefficients are taken from the torch/image library.
        Beware: these are more critical than you might think, as the
        monochromatic contrast can be surprisingly low.
        """
        if len(im.shape) < 3:
            return im
        return np.sum(im * [0.299, 0.587, 0.114], axis=2)


class NoNegativeRewardEnv(gym.RewardWrapper):
    """Clip reward in negative direction."""
    def __init__(self, env=None, neg_clip=0.0):
        super(NoNegativeRewardEnv, self).__init__(env)
        self.neg_clip = neg_clip

    def _reward(self, reward):
        new_reward = self.neg_clip if reward < self.neg_clip else reward
        return new_reward


class SkipEnv(gym.Wrapper):
    """Skip timesteps: repeat action, accumulate reward, take last obs."""
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self.skip = skip

    def _step(self, action):
        total_reward = 0
        for i in range(0, self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            info['steps'] = i + 1
            if done:
                break
        return obs, total_reward, done, info


class MarioEnv(gym.Wrapper):
    def __init__(self, env=None, tilesEnv=False):
        """Reset mario environment without actually restarting fceux everytime.
        This speeds up unrolling by approximately 10 times.
        """
        super(MarioEnv, self).__init__(env)
        self.resetCount = -1
        # reward is distance travelled. So normalize it with total distance
        # https://github.com/ppaquette/gym-super-mario/blob/master/ppaquette_gym_super_mario/lua/super-mario-bros.lua
        # However, we will not use this reward at all. It is only for completion.
        self.maxDistance = 3000.0
        self.tilesEnv = tilesEnv

    def _reset(self):
        if self.resetCount < 0:
            print('\nDoing hard mario fceux reset (40 seconds wait) !')
            sys.stdout.flush()
            self.env.reset()
            time.sleep(40)
        obs, _, _, info = self.env.step(7)  # take right once to start game
        if info.get('ignore', False):  # assuming this happens only in beginning
            self.resetCount = -1
            self.env.close()
            return self._reset()
        self.resetCount = info.get('iteration', -1)
        if self.tilesEnv:
            return obs
        return obs[24:-12,8:-8,:]

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        # print('info:', info)
        done = info['iteration'] > self.resetCount
        reward = float(reward)/self.maxDistance # note: we do not use this rewards at all.
        if self.tilesEnv:
            return obs, reward, done, info
        return obs[24:-12,8:-8,:], reward, done, info

    def _close(self):
        self.resetCount = -1
        return self.env.close()


class MakeEnvDynamic(gym.ObservationWrapper):
    """Make observation dynamic by adding noise"""
    def __init__(self, env=None, percentPad=5):
        super(MakeEnvDynamic, self).__init__(env)
        self.origShape = env.observation_space.shape
        newside = int(round(max(self.origShape[:-1])*100./(100.-percentPad)))
        self.newShape = [newside, newside, 3]
        self.observation_space = Box(0.0, 255.0, self.newShape)
        self.bottomIgnore = 20  # doom 20px bottom is useless
        self.ob = None

    def _observation(self, obs):
        imNoise = np.random.randint(0,256,self.newShape).astype(obs.dtype)
        imNoise[:self.origShape[0]-self.bottomIgnore, :self.origShape[1], :] = obs[:-self.bottomIgnore,:,:]
        self.ob = imNoise
        return imNoise

    # def render(self, mode='human', close=False):
    #     temp = self.env.render(mode, close)
    #     return self.ob
