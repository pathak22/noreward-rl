'''
Place this file in:
/home/pathak/projects/unsup-rl/unsuprl/local/lib/python2.7/site-packages/ppaquette_gym_doom/__init__.py
'''

from gym.envs.registration import register
from gym.scoreboard.registration import add_task, add_group
from .package_info import USERNAME
from .doom_env import DoomEnv, MetaDoomEnv
from .doom_basic import DoomBasicEnv
from .doom_corridor import DoomCorridorEnv
from .doom_defend_center import DoomDefendCenterEnv
from .doom_defend_line import DoomDefendLineEnv
from .doom_health_gathering import DoomHealthGatheringEnv
from .doom_my_way_home import DoomMyWayHomeEnv
from .doom_predict_position import DoomPredictPositionEnv
from .doom_take_cover import DoomTakeCoverEnv
from .doom_deathmatch import DoomDeathmatchEnv
from .doom_my_way_home_sparse import DoomMyWayHomeFixedEnv
from .doom_my_way_home_verySparse import DoomMyWayHomeFixed15Env

# Env registration
# ==========================

register(
    id='{}/meta-Doom-v0'.format(USERNAME),
    entry_point='{}_gym_doom:MetaDoomEnv'.format(USERNAME),
    timestep_limit=999999,
    reward_threshold=9000.0,
    kwargs={
        'average_over': 3,
        'passing_grade': 600,
        'min_tries_for_avg': 3
    },
)

register(
    id='{}/DoomBasic-v0'.format(USERNAME),
    entry_point='{}_gym_doom:DoomBasicEnv'.format(USERNAME),
    timestep_limit=10000,
    reward_threshold=10.0,
)

register(
    id='{}/DoomCorridor-v0'.format(USERNAME),
    entry_point='{}_gym_doom:DoomCorridorEnv'.format(USERNAME),
    timestep_limit=10000,
    reward_threshold=1000.0,
)

register(
    id='{}/DoomDefendCenter-v0'.format(USERNAME),
    entry_point='{}_gym_doom:DoomDefendCenterEnv'.format(USERNAME),
    timestep_limit=10000,
    reward_threshold=10.0,
)

register(
    id='{}/DoomDefendLine-v0'.format(USERNAME),
    entry_point='{}_gym_doom:DoomDefendLineEnv'.format(USERNAME),
    timestep_limit=10000,
    reward_threshold=15.0,
)

register(
    id='{}/DoomHealthGathering-v0'.format(USERNAME),
    entry_point='{}_gym_doom:DoomHealthGatheringEnv'.format(USERNAME),
    timestep_limit=10000,
    reward_threshold=1000.0,
)

register(
    id='{}/DoomMyWayHome-v0'.format(USERNAME),
    entry_point='{}_gym_doom:DoomMyWayHomeEnv'.format(USERNAME),
    timestep_limit=10000,
    reward_threshold=0.5,
)

register(
    id='{}/DoomPredictPosition-v0'.format(USERNAME),
    entry_point='{}_gym_doom:DoomPredictPositionEnv'.format(USERNAME),
    timestep_limit=10000,
    reward_threshold=0.5,
)

register(
    id='{}/DoomTakeCover-v0'.format(USERNAME),
    entry_point='{}_gym_doom:DoomTakeCoverEnv'.format(USERNAME),
    timestep_limit=10000,
    reward_threshold=750.0,
)

register(
    id='{}/DoomDeathmatch-v0'.format(USERNAME),
    entry_point='{}_gym_doom:DoomDeathmatchEnv'.format(USERNAME),
    timestep_limit=10000,
    reward_threshold=20.0,
)

register(
    id='{}/DoomMyWayHomeFixed-v0'.format(USERNAME),
    entry_point='{}_gym_doom:DoomMyWayHomeFixedEnv'.format(USERNAME),
    timestep_limit=10000,
    reward_threshold=0.5,
)

register(
    id='{}/DoomMyWayHomeFixed15-v0'.format(USERNAME),
    entry_point='{}_gym_doom:DoomMyWayHomeFixed15Env'.format(USERNAME),
    timestep_limit=10000,
    reward_threshold=0.5,
)

# Scoreboard registration
# ==========================
add_group(
    id= 'doom',
    name= 'Doom',
    description= 'Doom environments based on VizDoom.'
)

add_task(
    id='{}/meta-Doom-v0'.format(USERNAME),
    group='doom',
    summary='Mission #1 to #9 - Beat all 9 Doom missions.',
    description="""
This is a meta map that combines all 9 Doom levels.

Levels:
    - #0 Doom Basic
    - #1 Doom Corridor
    - #2 Doom DefendCenter
    - #3 Doom DefendLine
    - #4 Doom HealthGathering
    - #5 Doom MyWayHome
    - #6 Doom PredictPosition
    - #7 Doom TakeCover
    - #8 Doom Deathmatch
    - #9 Doom MyWayHomeFixed (customized)
    - #10 Doom MyWayHomeFixed15 (customized)

Goal: 9,000 points
    - Pass all levels

Scoring:
    - Each level score has been standardized on a scale of 0 to 1,000
    - The passing score for a level is 990 (99th percentile)
    - A bonus of 450 (50 * 9 levels) is given if all levels are passed
    - The score for a level is the average of the last 3 tries
"""
)

add_task(
    id='{}/DoomBasic-v0'.format(USERNAME),
    group='doom',
    summary='Mission #1 - Kill a single monster using your pistol.',
    description="""
This map is rectangular with gray walls, ceiling and floor.
You are spawned in the center of the longer wall, and a red
circular monster is spawned randomly on the opposite wall.
You need to kill the monster (one bullet is enough).

Goal: 10 points
    - Kill the monster in 3 secs with 1 shot

Rewards:
    - Plus 101 pts for killing the monster
    - Minus  5 pts for missing a shot
    - Minus  1 pts every 0.028 secs

Ends when:
    - Monster is dead
    - Player is dead
    - Timeout (10 seconds - 350 frames)

Allowed actions:
    - ATTACK
    - MOVE_RIGHT
    - MOVE_LEFT
"""
)

add_task(
    id='{}/DoomCorridor-v0'.format(USERNAME),
    group='doom',
    summary='Mission #2 - Run as fast as possible to grab a vest.',
    description="""
This map is designed to improve your navigation. There is a vest
at the end of the corridor, with 6 enemies (3 groups of 2). Your goal
is to get to the vest as soon as possible, without being killed.

Goal: 1,000 points
    - Reach the vest (or get very close to it)

Rewards:
    - Plus distance for getting closer to the vest
    - Minus distance for getting further from the vest
    - Minus 100 pts for getting killed

Ends when:
    - Player touches vest
    - Player is dead
    - Timeout (1 minutes - 2,100 frames)

Allowed actions:
    - ATTACK
    - MOVE_RIGHT
    - MOVE_LEFT
    - MOVE_FORWARD
    - TURN_RIGHT
    - TURN_LEFT
"""
)

add_task(
    id='{}/DoomDefendCenter-v0'.format(USERNAME),
    group='doom',
    summary='Mission #3 - Kill enemies coming at your from all sides.',
    description="""
This map is designed to teach you how to kill and how to stay alive.
You will also need to keep an eye on your ammunition level. You are only
rewarded for kills, so figure out how to stay alive.

The map is a circle with monsters. You are in the middle. Monsters will
respawn with additional health when killed. Kill as many as you can
before you run out of ammo.

Goal: 10 points
    - Kill 11 monsters (you have 26 ammo)

Rewards:
    - Plus 1 point for killing a monster
    - Minus 1 point for getting killed

Ends when:
    - Player is dead
    - Timeout (60 seconds - 2100 frames)

Allowed actions:
    - ATTACK
    - TURN_RIGHT
    - TURN_LEFT
"""
)

add_task(
    id='{}/DoomDefendLine-v0'.format(USERNAME),
    group='doom',
    summary='Mission #4 - Kill enemies on the other side of the room.',
    description="""
This map is designed to teach you how to kill and how to stay alive.
Your ammo will automatically replenish. You are only rewarded for kills,
so figure out how to stay alive.

The map is a rectangle with monsters on the other side. Monsters will
respawn with additional health when killed. Kill as many as you can
before they kill you. This map is harder than the previous.

Goal: 15 points
    - Kill 16 monsters

Rewards:
    - Plus 1 point for killing a monster
    - Minus 1 point for getting killed

Ends when:
    - Player is dead
    - Timeout (60 seconds - 2100 frames)

Allowed actions:
    - ATTACK
    - TURN_RIGHT
    - TURN_LEFT
"""
)

add_task(
    id='{}/DoomHealthGathering-v0'.format(USERNAME),
    group='doom',
    summary='Mission #5 - Learn to grad medkits to survive as long as possible.',
    description="""
This map is a guide on how to survive by collecting health packs.
It is a rectangle with green, acidic floor which hurts the player
periodically. There are also medkits spread around the map, and
additional kits will spawn at interval.

Goal: 1000 points
    - Stay alive long enough for approx. 30 secs

Rewards:
    - Plus 1 point every 0.028 secs
    - Minus 100 pts for dying

Ends when:
    - Player is dead
    - Timeout (60 seconds - 2,100 frames)

Allowed actions:
    - MOVE_FORWARD
    - TURN_RIGHT
    - TURN_LEFT
"""
)

add_task(
    id='{}/DoomMyWayHome-v0'.format(USERNAME),
    group='doom',
    summary='Mission #6 - Find the vest in one the 4 rooms.',
    description="""
This map is designed to improve navigational skills. It is a series of
interconnected rooms and 1 corridor with a dead end. Each room
has a separate color. There is a green vest in one of the room.
The vest is always in the same room. Player must find the vest.

Goal: 0.50 point
    - Find the vest

Rewards:
    - Plus 1 point for finding the vest
    - Minus 0.0001 point every 0.028 secs

Ends when:
    - Vest is found
    - Timeout (1 minutes - 2,100 frames)

Allowed actions:
    - MOVE_FORWARD
    - TURN_RIGHT
    - TURN_LEFT
"""
)

add_task(
    id='{}/DoomPredictPosition-v0'.format(USERNAME),
    group='doom',
    summary='Mission #7 - Learn how to kill an enemy with a rocket launcher.',
    description="""
This map is designed to train you on using a rocket launcher.
It is a rectangular map with a monster on the opposite side. You need
to use your rocket launcher to kill it. The rocket adds a delay between
the moment it is fired and the moment it reaches the other side of the room.
You need to predict the position of the monster to kill it.

Goal: 0.5 point
    - Kill the monster

Rewards:
    - Plus 1 point for killing the monster
    - Minus 0.0001 point every 0.028 secs

Ends when:
    - Monster is dead
    - Out of missile (you only have one)
    - Timeout (20 seconds - 700 frames)

Hint: Wait 1 sec for the missile launcher to load.

Allowed actions:
    - ATTACK
    - TURN_RIGHT
    - TURN_LEFT
"""
)

add_task(
    id='{}/DoomTakeCover-v0'.format(USERNAME),
    group='doom',
    summary='Mission #8 - Survive as long as possible with enemies shooting at you.',
    description="""
This map is to train you on the damage of incoming missiles.
It is a rectangular map with monsters firing missiles and fireballs
at you. You need to survive as long as possible.

Goal: 750 points
    - Survive for approx. 20 seconds

Rewards:
    - Plus 1 point every 0.028 secs

Ends when:
    - Player is dead (1 or 2 fireballs is enough)
    - Timeout (60 seconds - 2,100 frames)

Allowed actions:
    - MOVE_RIGHT
    - MOVE_LEFT
"""
)

add_task(
    id='{}/DoomDeathmatch-v0'.format(USERNAME),
    group='doom',
    summary='Mission #9 - Kill as many enemies as possible without being killed.',
    description="""
Kill as many monsters as possible without being killed.

Goal: 20 points
    - Kill 20 monsters

Rewards:
    - Plus 1 point for killing a monster

Ends when:
    - Player is dead
    - Timeout (3 minutes - 6,300 frames)

Allowed actions:
    - ALL
"""
)

add_task(
    id='{}/DoomMyWayHomeFixed-v0'.format(USERNAME),
    group='doom',
    summary='Mission #10 - Find the vest in one the 4 rooms.',
    description="""
This map is designed to improve navigational skills. It is a series of
interconnected rooms and 1 corridor with a dead end. Each room
has a separate color. There is a green vest in one of the room.
The vest is always in the same room. Player must find the vest.
You always start from fixed room (room no. 10 -- farthest).

Goal: 0.50 point
    - Find the vest

Rewards:
    - Plus 1 point for finding the vest
    - Minus 0.0001 point every 0.028 secs

Ends when:
    - Vest is found
    - Timeout (1 minutes - 2,100 frames)

Allowed actions:
    - MOVE_FORWARD
    - TURN_RIGHT
    - TURN_LEFT
"""
)

add_task(
    id='{}/DoomMyWayHomeFixed15-v0'.format(USERNAME),
    group='doom',
    summary='Mission #11 - Find the vest in one the 4 rooms.',
    description="""
This map is designed to improve navigational skills. It is a series of
interconnected rooms and 1 corridor with a dead end. Each room
has a separate color. There is a green vest in one of the room.
The vest is always in the same room. Player must find the vest.
You always start from fixed room (room no. 10 -- farthest).

Goal: 0.50 point
    - Find the vest

Rewards:
    - Plus 1 point for finding the vest
    - Minus 0.0001 point every 0.028 secs

Ends when:
    - Vest is found
    - Timeout (1 minutes - 2,100 frames)

Allowed actions:
    - MOVE_FORWARD
    - TURN_RIGHT
    - TURN_LEFT
"""
)
