# -*- coding: utf-8 -*-
# @Time    : 2019/11/26 19:37
# @Author  : Mrli
# @FileName: agents.py
# @Cnblogs ：https://nymrli.top/
import numpy as np
import time
import datetime
from core import Grid


class Agent(object):
    direct = ['↑', '↓', '←', '→']

    def __init__(self, env = None):
        self.env = env # 建立对环境对象的引用

    def getBestmove(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


class MinMaxAgent(Agent):
    ''''
    实现极小极大算法的agent
    '''
    def __init__(self, env=None, maxSearchTime=1000):
        super(MinMaxAgent, self).__init__(env=env)
        self.env = env # 建立对环境对象的引用
        self.maxSearchTime = maxSearchTime

    def search(self, data, depth, alpha, beta, positions, cutoffs, rootdepth) ->dict:
        '''
        Alpha-beta pruning
        root节点为depth层
        :param depth: 可以搜索的深度
        :param alpha: 搜索下限
        :param beta:  搜索上限
        :param positions:
        :param cutoffs: 被剪枝数
        :param rootdepth: root层是偶数层
        :return:
        '''
        bestMove = -1

        # grid = np.copy(self.env.data)
        grid = np.copy(data)
        if depth % 2 == rootdepth % 2:  # 偶数是自己 max层
            bestScore = alpha
            for direction in range(len(self.direct)):
                grid_t, same = Grid.rollup(grid, direction)
                if same: continue

                if Grid.is_win(grid_t): # 如果当前位置已经能够胜利的话，肯定这么走
                    return { 'move': direction, 'score': 10000, 'positions': positions, 'cutoffs': cutoffs }
                if depth == 0:  # 当深度为0时,即叶子节点时,判断其评估值
                    result =  { 'move': direction, 'score': Grid.evaluate(data) }
                else:
                    # 下探子节点
                    result = self.search(grid_t, depth - 1, bestScore, beta, positions, cutoffs, rootdepth)
                    if result['score'] > 9900:
                        # 从胜利的深度进行轻微的惩罚
                        result['score'] -= 10
                    # 记录子节点的位置
                    positions = result['positions']
                    # 记录子节点的被剪枝数
                    cutoffs = result['cutoffs']
                # 更新alpha值 ：alpha' == max(alpha, result['score'])
                if result['score'] > bestScore:
                    bestScore = result['score']
                    bestMove = direction

                if bestScore > beta:    # alpha > beta
                    # 被剪枝了
                    cutoffs += 1
                    return {'move': bestMove, 'score': beta, 'positions': positions, 'cutoffs': cutoffs}
        else:               # 奇数是对方 min层
            # TODO：
            bestScore = beta
            candidates = {}
            # cells = Grid.getAvailable(self.env.data)
            cells = Grid.getAvailable(data)
            scores = {2: {}, 4: {}}

            # 计算随机块生成的annoying 程度
            for k in scores.keys():
                for cell in cells:
                    row, col = cell
                    # 环境随机生成
                    grid[row][col] = k
                    # 计算annouing程度
                    # TODO: smoothness
                    # annoying = -Grid.smoothness(grid) + Grid.islands(grid)
                    annoying = Grid.islands(grid)
                    scores[k][cell] = annoying
                    # 复原
                    grid[row][col] = 0

            def getmax():
                num = []
                for v in scores.values():
                    for j in v.values():
                        num.append(j)
                return max(num)

            maxScore = getmax()
            # maxScore = max(max(scores.values(), key=lambda y: y[1]))
            for k in scores:
                for cell in scores[k]:
                    if scores[k][cell] == maxScore:
                        candidates[cell] = k

            for cell, value in candidates.items():
                row, col = cell
                grid[row][col] = value
                positions += 1
                result = self.search(grid, depth, alpha, bestScore, positions, cutoffs, depth)
                # 记录子节点的位置
                positions = result['positions']
                # 记录子节点的被剪枝数
                cutoffs = result['cutoffs']

                # min层, 更新beta
                if result['score'] < bestScore:
                    bestScore = result['score']

                if bestScore < alpha :
                    cutoffs += 1
                    return {'move': -1, 'score': alpha, 'positions': positions, 'cutoffs': cutoffs}

        return {'move': bestMove, 'score': bestScore, 'positions': positions, 'cutoffs': cutoffs}



    def iterativeDeep(self) ->int:
        '''
        限时进行迭代
        :return:
        '''
        getMS = lambda : int(round(time.time() * 1000))
        # start_ms = datetime.datetime.now().microsecond

        start_t = getMS()
        depth = 0
        move = -1
        now_t = getMS()
        # now_ms = datetime.datetime.now().microsecond
        while now_t - start_t < self.maxSearchTime :
        # while now_ms - start_ms < self.maxSearchTime :
            # best_move = self.search(depth, -np.inf, np.inf, 0, 0)
            res = self.search(self.env.data, depth, -10000, 10000, 0, 0, depth)
            best_move = res['move']
            if best_move == -1:
                return -1
            else:
                move = best_move
            depth += 1
            now_t = getMS()
            # now_ms = datetime.datetime.now().microsecond
        print("迭代了{}层".format(depth-1))
        return move

    def getBestmove(self) ->int:
        return self.iterativeDeep()

    def step(self):
        res = self.getBestmove()
        return res, self.direct[res]


class RandomAgent(Agent):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self, action_num: int=4, env=None):
        ''' Initilize the random agent

        Args:
            action_num (int): the size of the ouput action space
        '''
        super(RandomAgent, self).__init__(env=env)
        self.action_num = action_num
        self.env = env

    def getBestmove(self):
        return np.random.randint(0, self.action_num-1)

    def step(self):
        res = self.getBestmove()
        return res, self.direct[res]



    def eval_step(self, state):
        ''' Predict the action given the curent state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state)
