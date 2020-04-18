import json
from ws4py.client.threadedclient import WebSocketClient
import random
import multiprocessing

from agents import MinMaxAgent
from core import Game

class GamePlayer(WebSocketClient):

    def opened(self):
        # req = '{"event":"subscribe", "channel":"eth_usdt.deep"}'
        # self.send(req)
        print("连接成功")
        # 第一步随机下
        self.send('1')

        self.agent = MinMaxAgent(env=Game())

    def closed(self, code, reason=None):
        # print("Closed down:", code, reason)
        print("连接中断")

    def received_message(self, resp):
        '''
        接收数据:
        type 0表示结束，
             1表示进行，此时有新参数board: 4*4list
        resp = {
             type:1,
             board:[
                    [2,2,2,4,],
                    [2,2,2,4,],
                    [2,2,2,4,],
                    [2,2,2,4,]
                    ]
            }
        :param resp:
        :return:
        '''
        if '初始化游戏成功' in str(resp):
            pass
        else:
            resp = json.loads(str(resp))
            print(type(resp),resp)
            if isinstance(resp, list):
                self.send("{}".format(random.randint(0,3)))
            elif isinstance(resp, dict):
                status = resp.get('type')
                if status == "1":
                    board = resp.get('board')
                    # 随机下法
                    # move = random.randint(0,3)

                    # Minmax下法
                    self.agent.env.clone(board)
                    move, dir = self.agent.step()
                    print("---最佳策略为:{},{}---".format(dir, move))
                    self.send(str(move))
                else:
                    print("游戏结束")
                    self.send("grade")
            # elif isinstance(resp, int):
            #     if resp == 0:
            #         print("游戏结束")
            #         self.send("grade")


if __name__ == '__main__':
    # in_pipe, out_pipe = multiprocessing.Pipe(True)
    ws = None
    try:
        ws = GamePlayer('ws://127.0.0.1:23456/game')
        ws.connect()
        ws.run_forever()
    except KeyboardInterrupt:
        ws.close()
