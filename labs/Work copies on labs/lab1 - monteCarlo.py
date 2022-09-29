# imports
import math
from math import sqrt
import random
from typing import List, Tuple
import time
from copy import deepcopy  # world -> thought
import time
from unicodedata import name
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


class Human:
    def __init__(self, name):
        self.name = name

    def get_action(self, state: State):
        actiontoTake = -1

        human = int(input())
        actions = state.get_avail_actions()
        if human in actions:
            return human
        else:
            actiontoTake = random.choice(state.get_avail_actions())
            return actiontoTake

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
        self.ucb = 0
        self.c = 0.1
        self.explore = 0.1

    def nonTerminal(self):
        return (utility(self.state) == 0 and not self.state.is_over())

    def notFullyExpanded(self):
        actions = len(self.state.get_avail_actions())
        kids = len(self.children)

        not_kids_equal_actions = (kids < actions)
        return not_kids_equal_actions

    def qValue(self):
        p = self.parent
        if p != None:
            tmp = sqrt((2*math.log(p.visits+1)) / self.visits)
            q = self.wins/self.visits
            q = q + self.c*tmp
            return q
        return 0

    def bestChild(self, minmax):

        minimax_child = self.children[0]

        if (minmax % 2) == 0:
            for c in self.children:
                if c.q < minimax_child.q:
                    minimax_child = c
        else:
            for c in self.children:
                if c.q > minimax_child.q:
                    minimax_child = c

        return minimax_child

    def __repr__(self):
        return f"Wins {self.wins} visits {self.visits} q-value {round(self.q,3)} action {self.action}\n"


class MCTS(Agent):
    def __init__(self, name):
        super(MCTS, self).__init__(name)
        self.state = State()
        self.player = 1
        self.v_0 = None
        self.graph = None

        self.minmax = 1

    def get_action(self, state: State):
        self.state = state
        action = self.search()
        self.minmax += 1
        return action

    def getPlayerName(self):
        if (self.player == 1):
            return 'X'
        else:
            return 'O'

    def search(self):

        state_init = deepcopy(self.state)
        root = Node(state_init, None)

        start = time.time()
        decision_time = 2

        while (utility(self.state) == 0 and not state_init.is_over() and (time.time() < start + decision_time)):
            v_1 = self.treePolicy(root)
            stateThink = deepcopy(state_init)
            delta = self.defaultpolicy(stateThink)
            self.backup(v_1, delta)

        bestChild: Node = root.bestChild(self.minmax)

        return bestChild.action

    def treePolicy(self, node: Node):
        v = node

        while v.nonTerminal():
            if v.notFullyExpanded():
                return self.expand(v, v.state)
            else:
                v = v.bestChild(self.minmax)
        return v

    def expand(self, node: Node, state: State):

        action = None
        if len(node.untried) == 1:
            action = node.untried[0]
            node.untried = []
        else:
            action = node.untried.pop(0)

        new_state = deepcopy(state)
        new_state.put_action(action, Agent('X'))

        v_ = Node(new_state, node)
        v_.action = action

        node.children.append(v_)

        return v_

    def defaultpolicy(self, stateThink: State):

        s = deepcopy(stateThink)
        while (utility(s) == 0 and not s.is_over()):

            actions = s.get_avail_actions()
            action = actions.pop(random.randrange(0, len(actions)))
            s.put_action(action, Agent(self.getPlayerName()))
            self.player *= -1
        return utility(s)

    def backup(self, node: Node, delta):
        v = node

        while (v != None):

            v.visits += 1
            if delta > 0:
                v.wins += 1
            v.q = v.qValue()   # + delta)
            v = v.parent


agents = (Agent('O'), MCTS('X'))
game = Game(agents)
game.play()
