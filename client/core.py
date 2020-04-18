# -*- coding:utf-8 -*-
import time
import copy
import random
import numpy as np
from typing import Union

from config import *


class Grid(object):
    '''
    局面, 或许可以直接继承np.array
    '''

    @staticmethod
    def matric2vec(data: Union[np.matrix, np.array]):
        return data.flatten()

    @staticmethod
    def vec2matric(vec: np.array, size :tuple=(WINDOW_BLOCK_NUM, WINDOW_BLOCK_NUM)):
        return np.reshape(vec, newshape=size)


    @staticmethod
    def evaluate(data: np.array) ->float:
        # monoWeight = 0.0
        # islandWeight = 0.0
        smoothWeight = 0.1
        mono2Weight = 1.0
        emptyWeight = 2.7
        maxWeight = 1.0
        # return Grid.smoothness(data) * smoothWeight + \
        #        Grid.monotonicity2(data) * mono2Weight + \
        #        len(Grid.getAvailable(data)) * emptyWeight + \
        #        Grid.max(data) * maxWeight
        return len(Grid.getAvailable(data)) * emptyWeight + \
               Grid.max(data) * maxWeight

    @staticmethod
    def smoothness(data :np.array):
        pass
        return 1

    @staticmethod
    def monotonicity2(data :np.array):
        pass
        return 1

    @staticmethod
    def islands(data: np.array)-> int:
        '''
        counts the number of isolated groups.求连通分量数
        :param data:
        :return:
        '''
        m = np.zeros(shape=data.shape)
        group = 0

        def mark(i, j):
            m[i][j] = 1
            dir = {(0, -1), (0, 1), (-1, 0), (1, 0)}
            for d in dir:
                it = i + d[0]
                jt = j + d[1]
                if (it >= 0 and it < WINDOW_BLOCK_NUM) and (jt >= 0 and jt < WINDOW_BLOCK_NUM):
                    if m[it][jt] == 0 and data[it][jt] != 0:
                        mark(it, jt)

        for i in range(len(m)):
            for j in range(len(m[0])):
                if m[i][j] == 0 and data[i][j] != 0:
                    group += 1
                    mark(i, j)

        return group


    @staticmethod
    def init()-> np.array:
        data = np.zeros((WINDOW_BLOCK_NUM, WINDOW_BLOCK_NUM), dtype=np.int32)
        for i in range(2):
            row, col = np.random.randint(1, WINDOW_BLOCK_NUM, 2)
            data[row][col] = Grid.randomDig()
        return data


    @staticmethod
    def sum(data: Union[np.array, list]):
        return np.sum(data)

    @staticmethod
    def max(data: Union[np.array, list]):
        return np.max(data)

    @staticmethod
    def clone(data: Union[list, np.array]) ->np.array:
        return np.copy(data)

    @staticmethod
    def getAvailable(data: np.array) -> set:
        '''
        返回局面上空闲的位置
        :return: set
        '''
        available = set()
        # 检查棋盘上是否还有空位置
        for row in range(WINDOW_BLOCK_NUM):
            for col in range(WINDOW_BLOCK_NUM):
                if data[row][col] == 0: available.add((row, col))
        return available

    @staticmethod
    def newGrid(data: np.array) -> np.array:
        '''
        在当前局面上生成新的tile
        :param data: np.array
        :return: 新生成的data'
        '''
        available = Grid.getAvailable(data)
        if available:
            # 随机1-2块
            for i in range(random.randint(1, 2)):
                if available:
                    row, col = available.pop()
                    data[row][col] = Grid.randomDig()
            return data
        else:
            return data

    @staticmethod
    def randomDig() ->int:
        return 2 if random.random() <= GENERATE_THRESHOLD else 4

    @staticmethod
    def slideUpDown(data: np.array, up: bool):
        '''
        True向上滑动, False向下滑动
        :param up:
        :return: 移动后的局面, 移动前后是否一样, 移动的得分
        '''
        oldData = copy.deepcopy(data)
        score = 0
        for col in range(WINDOW_BLOCK_NUM):
            # 抽取一维非零向量
            cvl = [oldData[row][col] for row in range(WINDOW_BLOCK_NUM) if oldData[row][col]!=0]

            # 合并
            if len(cvl)>=2:
                score += Grid.merge(cvl, up)
            # 补零
            for i in range(WINDOW_BLOCK_NUM-len(cvl)):
                if up: cvl.append(0)
                else: cvl.insert(0,0)
            # 回填
            for row in range(WINDOW_BLOCK_NUM): oldData[row][col] = cvl[row]

        # return oldData, oldData==self.data, score
        return oldData, np.all(oldData==data), score

    @staticmethod
    def slideLeftRight(data: np.array, left: bool) ->(np.array, bool, int):
        '''
        True向左滑动, False向右滑动
        :param left:
        :return: 移动后的局面, 移动前后是否一样, 移动的得分
        '''
        oldData = copy.deepcopy(data)
        score = 0
        for row in range(WINDOW_BLOCK_NUM):
            rvl = [oldData[row][col] for col in range(WINDOW_BLOCK_NUM) if oldData[row][col]!=0]

            if len(rvl)>=2:
                score += Grid.merge(rvl, left)
            for i in range(WINDOW_BLOCK_NUM-len(rvl)):
                if left: rvl.append(0)
                else: rvl.insert(0,0)
            for col in range(WINDOW_BLOCK_NUM): oldData[row][col] = rvl[col]
        # return oldData, oldData==self.data, score
        return oldData, np.all(oldData==data), score

    @staticmethod
    def merge(vlist, direct) ->int:
        '''
        进行合并操作
        :param vlist:
        :param direct:
        :return: 合并得分
        '''
        score = 0
        if direct: #up or left
            i = 1
            while i<len(vlist):
                if vlist[i-1]==vlist[i]:
                    # 当两个块值相等，则删除一个，并让另一个值*2
                    del vlist[i]
                    vlist[i-1] *= 2
                    score += vlist[i-1]
                i += 1
        else:  #down or right
            i = len(vlist)-1
            while i>0:
                if vlist[i-1]==vlist[i]:
                    del vlist[i]
                    vlist[i-1] *= 2
                    score += vlist[i-1]
                i -= 1
        return score

    @staticmethod
    def rollup(data, dir: int):
        if dir == 0:
            grid_t, eq_tile, score = Grid.slideUpDown(data, True)
        elif dir == 1:
            grid_t, eq_tile, score = Grid.slideUpDown(data, False)
        elif dir == 2:
            grid_t, eq_tile, score = Grid.slideLeftRight(data, True)
        elif dir == 3:
            grid_t, eq_tile, score = Grid.slideLeftRight(data, False)
        else:
            raise Exception("输入类型错误")
        return grid_t, eq_tile

    @staticmethod
    def is_win(data: np.array) ->bool:
        '''
        判断游戏是否结束->赢了
        :return: 是否赢了
        '''
        return np.any((data==2048))


class Game(object):
    '''
    游戏环境
    '''
    def __init__(self, display=False, key_handle=None):
        if display:
            pass
            # self.window = Window2048()
            # self.keys = key.KeyStateHandler()
            # self.key_handle = key_handle
        self.initGame()

    def initGame(self) ->None:
        '''
        初始化
        :return:
        '''
        # 局面（是否开局就两块）
        self.data = Grid.init()
        # 是否结束
        self.over = False
        # Score
        self.score = 0

    def update(self) ->Grid:
        '''
        更新局面
        :return:
        '''
        self.data = Grid.newGrid(self.data)
        return self.data

    @property
    def state(self) ->Grid:
        '''
        返回当前局面信息
        :return:
        '''
        return self.data

    @property
    def isOver(self):
        return self.over

    @property
    def maxValue(self) ->int:
        return Grid.max(self.data)

    def slideUpDown(self, up: bool):
        '''
        True向上滑动, False向下滑动
        :param up:
        :return: 移动后的局面, 移动前后是否一样, 移动的得分
        '''
        oldData = copy.deepcopy(self.data)
        score = 0
        for col in range(WINDOW_BLOCK_NUM):
            # 抽取一维非零向量
            cvl = [oldData[row][col] for row in range(WINDOW_BLOCK_NUM) if oldData[row][col]!=0]

            # 合并
            if len(cvl)>=2:
                score += self.merge(cvl, up)
            # 补零
            for i in range(WINDOW_BLOCK_NUM-len(cvl)):
                if up: cvl.append(0)
                else: cvl.insert(0,0)
            # 回填
            for row in range(WINDOW_BLOCK_NUM): oldData[row][col] = cvl[row]

        # return oldData, oldData==self.data, score
        return oldData, np.all(oldData==self.data), score

    def slideLeftRight(self, left: bool) ->(np.array, bool, int):
        '''
        True向左滑动, False向右滑动
        :param left:
        :return: 移动后的局面, 移动前后是否一样, 移动的得分
        '''
        oldData = copy.deepcopy(self.data)
        # oldData = np.array(self.data)
        score = 0
        for row in range(WINDOW_BLOCK_NUM):
            rvl = [oldData[row][col] for col in range(WINDOW_BLOCK_NUM) if oldData[row][col]!=0]

            if len(rvl)>=2:
                score += self.merge(rvl, left)
            for i in range(WINDOW_BLOCK_NUM-len(rvl)):
                if left: rvl.append(0)
                else: rvl.insert(0,0)
            for col in range(WINDOW_BLOCK_NUM): oldData[row][col] = rvl[col]
        # return oldData, oldData==self.data, score
        return oldData, np.all(oldData==self.data), score

    def merge(self, vlist, direct) ->int:
        '''
        进行合并操作
        :param vlist:
        :param direct:
        :return: 合并得分
        '''
        score = 0
        if direct: #up or left
            i = 1
            while i<len(vlist):
                if vlist[i-1]==vlist[i]:
                    # 当两个块值相等，则删除一个，并让另一个值*2
                    del vlist[i]
                    vlist[i-1] *= 2
                    score += vlist[i-1]
                i += 1
        else:  #down or right
            i = len(vlist)-1
            while i>0:
                if vlist[i-1]==vlist[i]:
                    del vlist[i]
                    vlist[i-1] *= 2
                    score += vlist[i-1]
                i -= 1
        return score

    def game_over(self) ->bool:
        '''
        判断游戏是否结束->输了
        条件一：不能进行合并
        条件二： 没有空闲的格子
        :return:
        '''
        _,a,_ = self.slideUpDown(True)
        _,b,_ = self.slideUpDown(False)
        _,c,_ = self.slideLeftRight(True)
        _,d,_ = self.slideLeftRight(False)
        return a and b and c and d

    def game_win(self) ->bool:
        '''
        判断游戏是否结束->赢了
        :return: 是否赢了
        '''
        return np.any((self.data==2048))

    def inputDir(self, symbol:int) ->int:
        '''
        :param symbol: [0,1,2,3] 上下左右
        :return: 该动作的得分
        '''
        score = 0
        if symbol == 0:
            self.data, eq_tile, score = self.slideUpDown(True)
        elif symbol == 1:
            self.data, eq_tile, score = self.slideUpDown(False)
        elif symbol == 2:
            self.data, eq_tile, score = self.slideLeftRight(True)
        elif symbol == 3:
            self.data, eq_tile, score = self.slideLeftRight(False)
        self.update()
        self.score += score
        return score

    def step(self, input: Union[np.array, int]) ->bool:
        # 处理输入
        if isinstance(input, int):
            self.inputDir(input)
            if self.game_over() or self.game_win():
                self.over = True
            return True
        else:
            print("输入类型不正确")
            return False
            # raise Exception("输入类型不正确")


    def clone(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        if isinstance(data, list):
            self.data = np.array(data)
            # print("最好传入np.array类型的数据")

# if __name__ == '__main__':
#     env = Game()
#     agent = RandomAgent(action_num=4, env=env)
#     while not env.isOver:
#         act, move = agent.step()
#         print("当前下法：",move)
#         env.step(act)
#         print("当前棋盘:\n{}".format(env.state))
#         print("当前最大值:{}\n".format(env.maxValue) )
#     print("最终得分：", env.score)
