# imports
import math
from cmath import sqrt
import random
from typing import List, Tuple
import time
from copy import deepcopy  # world -> thought

# world and world model


class State:
    def __init__(self, cols=7, rows=6, win_req=4):
        self.board = [['.'] * cols for _ in range(rows)]
        self.heights = [1] * cols
        self.num_moves = 0
        self.win_req = win_req

    def get_avail_actions(self) -> List[int]:
        return [i for i in range(len(self.board[0])) if self.heights[i] <= len(self.board)]

    def put_action(self, action, agent):
        self.board[len(self.board) - self.heights[action]][action] = agent.name
        self.heights[action] += 1
        self.num_moves += 1

    def is_over(self):
        return self.num_moves >= len(self.board) * len(self.board[0])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        header = " ".join([str(i) for i in range(len(self.board[0]))])
        line = "".join(["-" for _ in range(len(header))])
        board = [[e for e in row] for row in self.board]
        board = '\n'.join([' '.join(row) for row in board])
        return '\n' + header + '\n' + line + '\n' + board + '\n'

# evaluate the utility of a state


def utility(state: 'State'):
    board = state.board
    n_cols = len(board[0]) - 1
    n_rows = len(board) - 1

    def diags_pos():
        """Get positive diagonals, going from bottom-left to top-right."""
        for di in ([(j, i - j) for j in range(n_cols)] for i in range(n_cols + n_rows - 1)):
            yield [board[i][j] for i, j in di if i >= 0 and j >= 0 and i < n_cols and j < n_rows]

    def diags_neg():
        """Get negative diagonals, going from top-left to bottom-right."""
        for di in ([(j, i - n_cols + j + 1) for j in range(n_cols)] for i in range(n_cols + n_rows - 1)):
            yield [board[i][j] for i, j in di if i >= 0 and j >= 0 and i < n_cols and j < n_rows]

    cols = list(map(list, list(zip(*board))))
    rows = board
    diags = list(diags_neg()) + list(diags_pos())
    lines = rows + cols + diags
    strings = ["".join(s) for s in lines]
    for string in strings:
        if 'OOOO' in string:
            return -1
        if 'XXXX' in string:
            return 1
    return 0

# parrent class for mcts, minmax, human, and any other idea for an agent you have


class Agent:
    def __init__(self, name: str):
        self.name: str = name

    def get_action(self, state: State):
        return random.choice(state.get_avail_actions())


# connecting states and agents
class Game:
    def __init__(self, agents: Tuple[Agent]):
        self.agents = agents
        self.state = State()

    def play(self):
        while utility(self.state) == 0 and not self.state.is_over():
            for agent in agents:
                if utility(self.state) == 0 and not self.state.is_over():
                    action = agent.get_action(self.state)
                    self.state.put_action(action, agent)
                    print(self.state)

# --------------------------------------------------------------------------------------------


class Node:
    def __init__(self, state: State, parent: 'Node' = None):
        #self.children: List['Nodes'] = []
        self.children = []
        self.parent: 'Node' = parent
        self.state: State = state
        self.action = None

        self.q = 0
        self.wins = 0
        self.visits = 0

    def nonTerminal(self):
        actions = len(self.state.get_avail_actions())
        return (actions > 0)

    def notFullyExpanded(self):
        actions = len(self.state.get_avail_actions())
        return ((len(self.children)) < (actions-1))

    def bestChild(self):
        if(self.parent != None):
            self.q = (self.wins/self.visits) + 0.01 * \
                sqrt(math.log(self.parent.visits)/self.visits)

        self.children.sort(key=lambda x: x.q)
        return self.children[0]


class MCTS(Agent):
    def __init__(self, name):
        super(MCTS, self).__init__(name)
        self.state = State()
        self.player = -1
        self.v_0 = Node(None)

    def get_action(self, state: State):
        self.state = state
        action = self.search()
        return action

    def getPlayerName(self):
        if (self.player == 1):
            return 'X'
        else:
            return 'O'

    def search(self):
        state_init = deepcopy(self.state)
        self.v_0 = Node(state_init, None)

        while (self.state.is_over):
            v_1 = self.treePolicy(self.v_0)
            stateThink = deepcopy(self.state)

            delta = self.defaultpolicy(stateThink)
            self.player *= -1
            self.backup(v_1, delta)

        return self.v_0.bestChild().action

    def treePolicy(self, node: Node):
        v = node

        while v.nonTerminal():

            if node.notFullyExpanded():
                v = self.expand(node, self.state)
            else:
                v = node.bestChild()
                break
        return v

    def expand(self, node: Node, state: State):
        print("Expand...")
        actions = state.get_avail_actions()
        action = actions.pop(0)

        stateThink = deepcopy(self.state)
        stateThink.put_action(action, Agent(self.getPlayerName()))

        v_ = Node(stateThink, node)
        v_.action = action
        node.children.append(v_)

        self.player *= -1

        print("Expand DONE")
        return v_

    def defaultpolicy(self, state: State):
        print("Defalut...")
        actions = state.get_avail_actions()
        stateThink = None

        while (len(actions) > 1):
            action = actions.pop(random.randrange(0, len(actions)))
            stateThink = deepcopy(self.state)
            self.player *= -1
            stateThink.put_action(action, Agent(self.getPlayerName()))

        print("Defalut DONE")

        return utility(stateThink)

    def backup(self, node: Node, delta):
        print("Backup...")
        v = node

        while v.parent is not None:

            N_v = v.parent
            N_v.visits += 1
            N_v.q = utility(N_v.state) + delta
            v = v.parent
        print("Backup DONE!")


agents = (Agent('O'), MCTS('X'))
game = Game(agents)
game.play()
