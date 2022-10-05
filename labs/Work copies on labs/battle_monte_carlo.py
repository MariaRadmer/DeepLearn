# imports
import math
from math import sqrt
import random
from typing import List, Tuple
import time
from copy import deepcopy  # world -> thought
import time
from unicodedata import name
from collections import defaultdict
from collections import deque
from operator import attrgetter
import numpy as np
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
            return q
        return 0

    def bestChild(self, minmax):

        minimax_child = self.children[0]

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
            return self.name
        else:
            if self.name == 'X':
                return 'O'
            else:
                return 'X'

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
        new_state.put_action(action, self)

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


class Node_Rikke:
    def __init__(self, state: State, parent=None, parent_action=0):
        self.children = []
        self.parent: Node_Rikke = parent
        self.parent_action = parent_action
        self.state: State = state
        self.num_visits = 0
        self.wins = 0
        self.q = 0.0
        self.untried_actions = deque(self.state.get_avail_actions())
        self.exploration = 1.4

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_not_terminal_node(self):
        return not self.state.is_over()

    def best_child(self):
        highest_q = 0
        best_child = self.children[0]
        for child in self.children[1:]:
            if child.q > highest_q:
                highest_q = child.q
                best_child = child
        return best_child

    def update_q(self, sim_num, curr_sims, num_moves):
        total_sims_after_ith_move = curr_sims+(num_moves*sim_num)
        # + self.exploration*math.sqrt(np.log(total_sims_after_ith_move+1)/self.num_visits)
        self.q = self.wins/self.num_visits

    def __repr__(self) -> str:
        return f"Wins {self.wins} visits{self.num_visits} q {self.q} action {self.parent_action}\n"


class MCTS_Rikke(Agent):
    def __init__(self, name):
        super(MCTS_Rikke, self).__init__(name)
        self.sim_num = 1000
        self.curr_sim = 0
        self.num_move = 0

    def get_action(self, state: State):
        self.num_move += 1
        new_state = deepcopy(state)
        root_node = Node_Rikke(new_state)
        for i in range(self.sim_num):
            self.curr_sim = i
            node = self.select(root_node)
            sim_state = deepcopy(node.state)
            theta = self.simulate(sim_state)  # theta = result of simulation
            self.backpropagate(node, theta)
        return root_node.best_child().parent_action

    def select(self, root_node: Node_Rikke) -> Node_Rikke:  # tree policy
        node = root_node
        while node.is_not_terminal_node():
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.best_child()
        return node

    def expand(self, parent: Node_Rikke) -> Node_Rikke:
        action = parent.untried_actions.popleft()
        child_state = deepcopy(parent.state)
        child_state.put_action(action, self)
        child = Node_Rikke(child_state, parent, action)
        parent.children.append(child)
        return child

    def simulate(self, state: State):
        sim_state = state
        player = self.name
        while utility(sim_state) == 0 and not sim_state.is_over():
            action = random.choice(sim_state.get_avail_actions())
            sim_state = deepcopy(sim_state)
            sim_state.put_action(action, Agent(player))
            if player == 'X':
                player = 'O'
            else:
                player = 'X'
        return utility(sim_state)

    def backpropagate(self, node: Node_Rikke, theta):
        node.num_visits += 1
        if theta == -1:
            node.wins += 1
        node.update_q(self.sim_num, self.curr_sim, self.num_move)
        if node.parent is not None:
            self.backpropagate(node.parent, theta)


agents = (MCTS('O'), MCTS_Rikke('X'))
game = Game(agents)
game.play()
