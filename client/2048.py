# -*- coding:utf-8 -*-
import random
import time
from core import Game, Grid
from agents import RandomAgent, MinMaxAgent


if __name__ == '__main__':
    env = Game()
    # agent = RandomAgent(action_num=4, env=env)
    agent = MinMaxAgent(env=env, maxSearchTime=100)

    while not env.isOver:
        act, move = agent.step()
        print("当前下法：",move)
        env.step(act)
        print("当前棋盘:\n{}".format(env.state))
        print("当前最大值:{}\n".format(env.maxValue) )
    print("最终得分：", env.score)
