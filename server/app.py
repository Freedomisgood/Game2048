# -*- coding: utf-8 -*-
# @Time    : 2019/11/28 17:40
# @Author  : Mrli
# @FileName: app.py
# @Cnblogs ：https://nymrli.top/

# !/usr/bin/env python
# -*- coding:utf-8 -*-
# app.py
import uuid
import json
import tornado.ioloop
import tornado.web
import tornado.websocket
from core import Game
import numpy as np


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')


class ChatHandler(tornado.websocket.WebSocketHandler):
    # 用户存储当前聊天室用户
    waiters = set()
    # 用于存储历时消息
    messages = []

    def open(self):
        """
        客户端连接成功时，自动执行
        :return:
        """
        ChatHandler.waiters.add(self)
        uid = str(uuid.uuid4())
        self.write_message(uid)

        for msg in ChatHandler.messages:
            content = self.render_string('message.html', **msg)
            self.write_message(content)

    def on_message(self, message):
        """
        客户端连发送消息时，自动执行
        :param message:
        :return:
        """

        msg = json.loads(message)
        ChatHandler.messages.append(message)

        # 向当前在线的每个用户发送最新的消息界面
        for client in ChatHandler.waiters:
            content = client.render_string('message.html', **msg)
            client.write_message(content)

    def on_close(self):
        """
        客户端关闭连接时，，自动执行
        :return:
        """
        ChatHandler.waiters.remove(self)


class GameHandler(tornado.websocket.WebSocketHandler):
    # 用户存储当前聊天室用户
    waiters = set()
    # 用于存储历时消息
    messages = []

    g = Game()

    def check_origin(self, origin):
        return True

    def open(self):
        """
        客户端连接成功时，自动执行
        :return:
        """

        GameHandler.waiters.add(self)
        uid = str(uuid.uuid4())
        if uid not in GameHandler.waiters:
            GameHandler.g.initGame()
            print('{}初始化游戏成功'.format(uid))
            self.write_message('{}初始化游戏成功'.format(uid))


    def on_message(self, message):
        """
        客户端连发送消息时，自动执行
        grade: 查看比赛得分
        :param message: [0,1,2,3]
        :return:
        """
        print("message", type(message), message) # message -> str
        if message == 'grade':
            print(GameHandler.g.score)
        else:
            msg = json.loads(message)
            if msg in range(0,4):
                ans = GameHandler.g.step(msg)
                data = GameHandler.g.state.tolist()
                print(data)
                if GameHandler.g.isOver:
                    send_data = {'type':'0'}
                    self.write_message(json.dumps(send_data))
                else:
                    send_data = {'type':'1', 'board':data}
                    self.write_message(send_data)


    def on_close(self):
        """
        客户端关闭连接时，，自动执行
        :return:
        """
        ChatHandler.waiters.remove(self)


def run():
    settings = {
        'template_path': 'templates',
        'static_path': 'static',
    }
    application = tornado.web.Application([
        (r"/", IndexHandler),
        (r"/chat", ChatHandler),
        (r"/game", GameHandler),
    ], **settings)
    application.listen(23456)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    run()
