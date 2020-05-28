from random import shuffle
import numpy as np


class HtmfGame:
    """ Minimal version of Hey, That's My Fish! for 2 players """

    TILE_SET = [1]*30 + [2]*20 + [3]*10
    # TILE_SET = [1]*100
    PENGUIN_COUNT = 4
    PHASES = [0, 1, 2]   # Setup - Movement - End

    def __init__(self):
        self.map = None
        self.scores = []
        self.tiles = []  # Needed for tiebreaker
        self.next_player = None
        self.phase = 0
        self.reset()

    def reset(self):
        self.scores = [0, 0]
        self.tiles = [0, 0]
        self.next_player = 0
        self.phase = 0
        self.map = IceFloeMap(self.TILE_SET)

    def place(self, destination):
        """ Returns tuple of next phase and next player """

        if self.phase != 0:
            raise ValueError('Not setup phase')

        # If tile not taken
        if self.map[destination] not in [1, 2, 3]:
            raise ValueError('Invalid destination for penguin')

        self.scores[self.next_player] += self.map[destination]
        self.tiles[self.next_player] += 1
        self.map[destination] = self.next_player + 5

        self.next_player = (self.next_player + 1) % 2

        # If all placed for each player
        if self.map.flat().count(5) >= self.PENGUIN_COUNT and self.map.flat().count(6) >= self.PENGUIN_COUNT:
            self.phase = 1

        return self.phase, self.next_player

    def move(self, target, destination):
        """ Returns player number of winner if game is finished, None otherwise
            Raises ValueError if action is invalid """

        if self.phase != 1:
            raise ValueError('Not move phase')

        # Check if valid move, doesn't move over or into gaps or other penguins
        path = self.map.path(target, destination)
        if not all([x in [1, 2, 3] for x in path]):
            raise ValueError('Invalid path')

        # Check if valid player making move
        if self.map[target] != self.next_player + 5:
            raise ValueError('Not that player\'s penguin')

        self.scores[self.next_player] += self.map[destination]
        self.tiles[self.next_player] += 1
        self.map[destination] = self.map[target]
        self.map[target] = 0

        self.next_player = (self.next_player + 1) % 2
        # Skip next player if they have no moves
        if not self.get_valid_moves():
            self.next_player = (self.next_player + 1) % 2
            # Neither player has moves means game is over
            if not self.get_valid_moves():
                self.phase = 2

        return self.phase, self.next_player

    def get_valid_placements(self):
        return self.map.find([1])

    # For each available penguin return list of movable positions for current player
    def get_valid_moves(self):
        moves = {}
        directions = [(1, 0), (-1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
        for x, y in self.map.find([self.next_player + 5]):
            peng_moves = []
            for dirx, diry in directions:
                for i in range(1, 8):
                    move = (x + dirx*i, y + diry*i)
                    if move not in self.map:
                        break
                    if self.map[move] in [1, 2, 3]:
                        peng_moves.append(move)
                    else:
                        break
            if peng_moves:
                moves[(x, y)] = peng_moves
        return moves

    def get_reward(self, player):
        if self.scores[0] > self.scores[1]:
            reward = 1.0
        elif self.scores[0] < self.scores[1]:
            reward = 0.0
        else:
            if self.tiles[0] > self.tiles[1]:
                reward = 1.0
            elif self.tiles[0] < self.tiles[1]:
                reward = 0.0
            else:
                reward = .5  # Draw
        return (1.0 - reward) if player else reward

    def get_score(self):
        return self.scores, self.tiles

    def render_map(self):
        mapping = {
            5: '$',
            6: '@',
            0: ' '
        }
        for y in range(7, -1, -1):
            print('%d -----' % y, end='')
            if y % 2 == 1:
                print('  ', end='')
            for x in range(0, 11):
                if (x, y) in self.map:
                    val = self.map[(x, y)]
                    print('   %s' % mapping.get(val, val), end='')
            print('')
        print('  /   /   /   /   /   /   /   /   /   /')
        print(' 1   2   3   4   5   6   7   8   9   10')

    # Returns data object containing all state information in a way the network can use it
    def get_state(self, player=None):
        if not player:
            player = self.next_player

        state = {
            'fish': self.map(mapping={1: .33333, 2: .66667, 3: 1.0}, default=0.0),
            'penguins': self.map(mapping={(player+5): 0.0, (player+1)%2+5: 1.0}, default=.5),
            'score': list(reversed([x/100 for x in self.scores])) if player else [x/100 for x in self.scores],
            'tiles': list(reversed([x/60 for x in self.tiles])) if player else [x/60 for x in self.tiles],
            'phase': self.phase,
            'placements_map': self.map(mapping={1: 1.0}, default=0.0),  # Can only place on one-fish
            'placements': self.get_valid_placements(),
            'moves': self.get_valid_moves(),
            'player_id': self.next_player
        }
        return state


class IceFloeMap:
    """ Defines a hex grid of a size needed for Hey, That's My Fish
          Supports indexing with coordinate tuples
          Membership tests whether provided coordinates are in play area
    """

    BOUNDS = {
            0: (6, 7),
            1: (4, 7),
            2: (2, 7),
            3: (0, 7),
            4: (0, 7),
            5: (0, 7),
            6: (0, 7),
            7: (0, 6),
            8: (0, 4),
            9: (0, 2),
            10: (0, 0)
        }
    DEFAULT_VALUE = 0
    X_SIZE = 11
    Y_SIZE = 8

    def __init__(self, initial_values):
        if len(initial_values) != 60:
            raise ValueError
        shuffle(initial_values)
        value_iter = iter(initial_values)

        self._map = [
            [next(value_iter) if (x, y) in self else self.DEFAULT_VALUE
             for y in range(self.Y_SIZE)]
            for x in range(self.X_SIZE)]

    # Returns whether a coordinate is within the map
    def __contains__(self, position):
        return self.contains(position)

    def contains(self, position):
        x, y = position
        return x in self.BOUNDS and self.BOUNDS[x][0] <= y <= self.BOUNDS[x][1]

    def __getitem__(self, position):
        return self._map[position[0]][position[1]]

    def __setitem__(self, position, value):
        if position not in self:
            raise IndexError

        self._map[position[0]][position[1]] = value

    # Returns whole map as numpy array with the mapping of values applied
    def __call__(self, mapping=None, default=0.0):
        if not mapping:
            mapping = {}

        arr = np.zeros((self.X_SIZE, self.Y_SIZE))
        for y in range(self.Y_SIZE):
            for x in range(self.X_SIZE):
                if self._map[x][y] in mapping:
                    arr[x][y] = mapping[self._map[x][y]]
                else:
                    arr[x][y] = default

        return arr

    # Returns flattened list of values in valid locations
    def flat(self):
        return [self._map[x][y] for y in range(self.Y_SIZE) for x in range(self.X_SIZE) if (x, y) in self]

    # Returns coordinates for all locations in values
    def find(self, values):
        return [(x, y) for y in range(self.Y_SIZE) for x in range(self.X_SIZE) if self._map[x][y] in values]

    # Returns array of values on a path, excludes start
    def path(self, start, end):
        if start not in self or end not in self:
            raise ValueError('Not a valid path')

        p = []
        diffx = end[0] - start[0]
        diffy = end[1] - start[1]

        if diffx == 0:
            y = start[1]
            for x in range(start[0]+1, end[0]+1, 1 if diffx >= 0 else -1):
                p.append(self._map[x][y])
        elif diffy == 0:
            x = start[0]
            for y in range(start[1]+1, end[1]+1, 1 if diffy >= 0 else -1):
                p.append(self._map[x][y])
        elif diffx + diffy == 0:
            x = start[0]
            y = start[1]
            for d in range(1, diffx+1):
                if diffx > 0:
                    p.append(self._map[x+d][y-d])
                else:
                    p.append(self._map[x-d][y+d])
        else:
            raise ValueError('Not a valid path')

        return p
