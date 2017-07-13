'''
Script to test if mario installation works fine. It
displays the game play simultaneously.
'''

from __future__ import print_function
import gym, universe
import env_wrapper
import ppaquette_gym_super_mario
from ppaquette_gym_super_mario import wrappers
import numpy as np
import time
from PIL import Image
import utils

outputdir = './gray42/'
env_id = 'ppaquette/SuperMarioBros-1-1-v0'
env = gym.make(env_id)
modewrapper = wrappers.SetPlayingMode('algo')
acwrapper = wrappers.ToDiscrete()
env = modewrapper(acwrapper(env))
env = env_wrapper.MarioEnv(env)

freshape = fshape = (42, 42)
env.seed(None)
env = env_wrapper.NoNegativeRewardEnv(env)
env = env_wrapper.DQNObsEnv(env, shape=freshape)
env = env_wrapper.BufferedObsEnv(env, n=4, skip=1, shape=fshape, channel_last=True)
env = env_wrapper.EltwiseScaleObsEnv(env)

start = time.time()
episodes = 0
maxepisodes = 1
env.reset()
imCount = 1
utils.mkdir_p(outputdir + '/ep_%02d/'%(episodes+1))
while(1):
    obs, reward, done, info = env.step(env.action_space.sample())
    Image.fromarray((255*obs).astype('uint8')).save(outputdir + '/ep_%02d/%06d.jpg'%(episodes+1,imCount))
    imCount += 1
    if done:
        episodes += 1
        print('Ep: %d, Distance: %d'%(episodes, info['distance']))
        if episodes >= maxepisodes:
            break
        env.reset()
        imCount = 1
        utils.mkdir_p(outputdir + '/ep_%02d/'%(episodes+1))
end = time.time()
print('\nTotal Time spent: %0.2f seconds'% (end-start))
env.close()
print('Done!')
exit(1)
