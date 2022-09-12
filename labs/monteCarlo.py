# imports
import math
from math import sqrt
import random
from typing import List, Tuple
import time
from copy import deepcopy  # world -> thought
import time
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
    def __init__(self, state: State, parent):
        #self.children: List['Nodes'] = []
        self.children = []
        self.parent = parent
        self.state: State = state
        self.action = None
        self.tried = False
        self.untried = state.get_avail_actions()

        self.q: float = 0
        self.wins: int = 0
        self.visits: int = 0

    def nonTerminal(self):
        return not self.state.is_over()

    def notFullyExpanded(self):
        actions = len(self.state.get_avail_actions())
        return ((len(self.children)) < (actions-1))

    # this does not work
    def qValue(self):
        p: Node = self.parent
        if p != None:
            print(p.visits)

            tmp = (2*math.log(p.visits + 1)) ** .5
            self.q = (self.wins/self.visits) + 0.001 * tmp
            return self.q
        return 0

    # this does not work
    def bestChild(self):
        self.children.sort(key=lambda x: x.q, reverse=True)
        print(self.children)
        return self.children[0]

    def __repr__(self):
        return str(self.wins)


class MCTS(Agent):
    def __init__(self, name):
        super(MCTS, self).__init__(name)
        self.state = State()
        self.player = 1
        self.v_0 = None
        self.graph = None

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

        start = time.time()
        decision_time = 1.5

        while (utility(self.state) == 0 and not state_init.is_over() and (time.time() < start + decision_time)):
            v_1 = self.treePolicy(self.v_0)
            stateThink = deepcopy(self.state)

            delta = self.defaultpolicy(stateThink)

            self.backup(v_1, delta)

        bestChild: Node = self.v_0.bestChild()
        return bestChild.action

    def treePolicy(self, node: Node):
        v = node

        while v.nonTerminal():
            if node.notFullyExpanded():
                tmp = deepcopy(node)
                v.children.append(self.expand(tmp, self.state))
            else:
                v = node.bestChild()
                break
        return v

    def expand(self, node: Node, state: State):
        action = node.untried.pop(0)

        stateThink = deepcopy(self.state)
        stateThink.put_action(action, Agent('X'))

        v_ = Node(stateThink, node)
        v_.action = action
        node.children.append(v_)

        return v_

    def defaultpolicy(self, state: State):

        stateThink = deepcopy(self.state)

        while (not stateThink.is_over()):
            actions = stateThink.get_avail_actions()
            action = actions.pop(random.randrange(0, len(actions)))

            stateThink.put_action(action, Agent(self.getPlayerName()))
            self.player *= -1

        return utility(stateThink)

    def backup(self, node: Node, delta):
        v = node

        while (v != None):

            v.visits += 1
            if delta > 0:
                v.wins += 1

            v.q = v.qValue() + delta  # v.qValue() + delta # this does not work
            v = v.parent


agents = (Agent('O'), MCTS('X'))
game = Game(agents)
game.play()
