'''
Place this file in:
/home/pathak/projects/unsup-rl/unsuprl/local/lib/python2.7/site-packages/ppaquette_gym_doom/wrappers/action_space.py
'''

import gym

# Constants
NUM_ACTIONS = 43
ALLOWED_ACTIONS = [
    [0, 10, 11],                                # 0 - Basic
    [0, 10, 11, 13, 14, 15],                    # 1 - Corridor
    [0, 14, 15],                                # 2 - DefendCenter
    [0, 14, 15],                                # 3 - DefendLine
    [13, 14, 15],                               # 4 - HealthGathering
    [13, 14, 15],                               # 5 - MyWayHome
    [0, 14, 15],                                # 6 - PredictPosition
    [10, 11],                                   # 7 - TakeCover
    [x for x in range(NUM_ACTIONS) if x != 33], # 8 - Deathmatch
    [13, 14, 15],                               # 9 - MyWayHomeFixed
    [13, 14, 15],                               # 10 - MyWayHomeFixed15
]

__all__ = [ 'ToDiscrete', 'ToBox' ]

def ToDiscrete(config):
    # Config can be 'minimal', 'constant-7', 'constant-17', 'full'

    class ToDiscreteWrapper(gym.Wrapper):
        """
            Doom wrapper to convert MultiDiscrete action space to Discrete

            config:
                - minimal - Will only use the levels' allowed actions (+ NOOP)
                - constant-7 - Will use the 7 minimum actions (+NOOP) to complete all levels
                - constant-17 - Will use the 17 most common actions (+NOOP) to complete all levels
                - full - Will use all available actions (+ NOOP)

            list of commands:
                - minimal:
                    Basic:              NOOP, ATTACK, MOVE_RIGHT, MOVE_LEFT
                    Corridor:           NOOP, ATTACK, MOVE_RIGHT, MOVE_LEFT, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
                    DefendCenter        NOOP, ATTACK, TURN_RIGHT, TURN_LEFT
                    DefendLine:         NOOP, ATTACK, TURN_RIGHT, TURN_LEFT
                    HealthGathering:    NOOP, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
                    MyWayHome:          NOOP, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
                    PredictPosition:    NOOP, ATTACK, TURN_RIGHT, TURN_LEFT
                    TakeCover:          NOOP, MOVE_RIGHT, MOVE_LEFT
                    Deathmatch:         NOOP, ALL COMMANDS (Deltas are limited to [0,1] range and will not work properly)

                - constant-7: NOOP, ATTACK, MOVE_RIGHT, MOVE_LEFT, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT, SELECT_NEXT_WEAPON

                - constant-17: NOOP, ATTACK, JUMP, CROUCH, TURN180, RELOAD, SPEED, STRAFE, MOVE_RIGHT, MOVE_LEFT, MOVE_BACKWARD
                                MOVE_FORWARD, TURN_RIGHT, TURN_LEFT, LOOK_UP, LOOK_DOWN, SELECT_NEXT_WEAPON, SELECT_PREV_WEAPON
        """
        def __init__(self, env):
            super(ToDiscreteWrapper, self).__init__(env)
            if config == 'minimal':
                allowed_actions = ALLOWED_ACTIONS[self.unwrapped.level]
            elif config == 'constant-7':
                allowed_actions = [0, 10, 11, 13, 14, 15, 31]
            elif config == 'constant-17':
                allowed_actions = [0, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32]
            elif config == 'full':
                allowed_actions = None
            else:
                raise gym.error.Error('Invalid configuration. Valid options are "minimal", "constant-7", "constant-17", "full"')
            self.action_space = gym.spaces.multi_discrete.DiscreteToMultiDiscrete(self.action_space, allowed_actions)
        def _step(self, action):
            return self.env._step(self.action_space(action))

    return ToDiscreteWrapper

def ToBox(config):
    # Config can be 'minimal', 'constant-7', 'constant-17', 'full'

    class ToBoxWrapper(gym.Wrapper):
        """
            Doom wrapper to convert MultiDiscrete action space to Box

            config:
                - minimal - Will only use the levels' allowed actions
                - constant-7 - Will use the 7 minimum actions to complete all levels
                - constant-17 - Will use the 17 most common actions to complete all levels
                - full - Will use all available actions

            list of commands:
                - minimal:
                    Basic:              ATTACK, MOVE_RIGHT, MOVE_LEFT
                    Corridor:           ATTACK, MOVE_RIGHT, MOVE_LEFT, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
                    DefendCenter        ATTACK, TURN_RIGHT, TURN_LEFT
                    DefendLine:         ATTACK, TURN_RIGHT, TURN_LEFT
                    HealthGathering:    MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
                    MyWayHome:          MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
                    PredictPosition:    ATTACK, TURN_RIGHT, TURN_LEFT
                    TakeCover:          MOVE_RIGHT, MOVE_LEFT
                    Deathmatch:         ALL COMMANDS

                - constant-7: ATTACK, MOVE_RIGHT, MOVE_LEFT, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT, SELECT_NEXT_WEAPON

                - constant-17:  ATTACK, JUMP, CROUCH, TURN180, RELOAD, SPEED, STRAFE, MOVE_RIGHT, MOVE_LEFT, MOVE_BACKWARD
                                MOVE_FORWARD, TURN_RIGHT, TURN_LEFT, LOOK_UP, LOOK_DOWN, SELECT_NEXT_WEAPON, SELECT_PREV_WEAPON
        """
        def __init__(self, env):
            super(ToBoxWrapper, self).__init__(env)
            if config == 'minimal':
                allowed_actions = ALLOWED_ACTIONS[self.unwrapped.level]
            elif config == 'constant-7':
                allowed_actions = [0, 10, 11, 13, 14, 15, 31]
            elif config == 'constant-17':
                allowed_actions = [0, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32]
            elif config == 'full':
                allowed_actions = None
            else:
                raise gym.error.Error('Invalid configuration. Valid options are "minimal", "constant-7", "constant-17", "full"')
            self.action_space = gym.spaces.multi_discrete.BoxToMultiDiscrete(self.action_space, allowed_actions)
        def _step(self, action):
            return self.env._step(self.action_space(action))

    return ToBoxWrapper
