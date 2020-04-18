# Game2048
运用MaxMin算法实现的2048agent，仿照https://github.com/ovolve/2048-AI的js版本转写成的Python版本

## 阶段1

搭建2048环境，构建RandomAgent。
运行:`python 2048.py`

## 阶段2

实现Minmax算法，MinmaxAgent

## 阶段3

完成websocket通信，先运行`server/app.py`再运行`client/app.py`

```
G:.
├───client
│   ├───node_modules
│   │   ├───async-limiter
│   │   └───ws
│   │       └───lib
│   ├───utils
│   └───__pycache__
├───server
│   ├───static
│   ├───templates
│   └───__pycache__
└───__pycache__
```

