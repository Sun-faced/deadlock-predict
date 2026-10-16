"""Column positions and category sizes shared by the match models.

The dataset stores each player group in team order: six players from side 0,
then six from side 1. Keep these slices in one place so the models read the
same columns when the dataset layout changes.
"""

TEAMS_PER_MATCH = 2
PLAYERS_PER_TEAM = 6
PLAYERS_PER_MATCH = TEAMS_PER_MATCH * PLAYERS_PER_TEAM
SIDE0_PLAYERS = slice(0, PLAYERS_PER_TEAM)
SIDE1_PLAYERS = slice(PLAYERS_PER_TEAM, PLAYERS_PER_MATCH)

HERO_ID_COUNT = 256  # hero IDs are used directly as embedding/one-hot positions
LANE_CODES = (1, 4, 6)  # lane values currently recognized by the models
LANE_POSITION_COUNT = len(LANE_CODES) + 1  # extra position for unknown lanes
PARTY_SIZE_COUNT = PLAYERS_PER_TEAM + 1  # stored party sizes range from 0 to 6
EXPERIENCE_VALUE_COUNT = 2  # prior games and prior wins on this hero
FLAG_COUNT = 3  # new-player pool, low-priority pool, high-skill parties

HERO_INPUTS = slice(0, PLAYERS_PER_MATCH)
LANE_INPUTS = slice(HERO_INPUTS.stop, HERO_INPUTS.stop + PLAYERS_PER_MATCH)
PARTY_INPUTS = slice(LANE_INPUTS.stop, LANE_INPUTS.stop + PLAYERS_PER_MATCH)
GAMES_INPUTS = slice(PARTY_INPUTS.stop, PARTY_INPUTS.stop + PLAYERS_PER_MATCH)
WINS_INPUTS = slice(GAMES_INPUTS.stop, GAMES_INPUTS.stop + PLAYERS_PER_MATCH)
FLAG_INPUTS = slice(WINS_INPUTS.stop, WINS_INPUTS.stop + FLAG_COUNT)
INPUT_COUNT = FLAG_INPUTS.stop

WIN_OUTPUT = 0
TIME_OUTPUT = WIN_OUTPUT + 1
KILLS_OUTPUTS = slice(TIME_OUTPUT + 1, TIME_OUTPUT + 1 + TEAMS_PER_MATCH)
BOSSES_OUTPUTS = slice(KILLS_OUTPUTS.stop, KILLS_OUTPUTS.stop + TEAMS_PER_MATCH)
OUTPUT_COUNT = BOSSES_OUTPUTS.stop

SECONDS_PER_MINUTE = 60
EXPERIENCE_MATCH_SCALE = 100
