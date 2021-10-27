import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from Board.board import Board
from Net.net import net
from collections import namedtuple

# マスの状態
EMPTY = 0 # 空きマス
WHITE = -1 # 白石
BLACK = 1 # 黒石
WALL = 2 # 壁

# 手の表現
IN_ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
IN_NUMBER = ['1', '2', '3', '4', '5', '6', '7', '8']
 
# ボードのサイズ
BOARD_SIZE = 6

# 経験保存時の名前
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# 経験を保存するメモリクラス
class ExperienceMemory():
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0
        
    def push(self, state, action, state_next, reward):
        ''' transition をメモリに保存する '''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index+1) % self.capacity
        
    def sample(self, batch_size):
        ''' batch_size 分だけランダムに保存内容を取り出す '''
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# class Board() のRawBoardをIN = [0] * (BOARD_SIZE * BOARD_SIZE)の形に変換
def trans(RawBoard_t):
    count = 0
    out = [0] * (BOARD_SIZE * BOARD_SIZE)
    for j in range(0,(BOARD_SIZE + 2)):
        for i in range(0,(BOARD_SIZE + 2)):
            if RawBoard_t[j,i] != 2:
                out[count] = RawBoard_t[j,i]
                count += 1
    return out

# 参考文献：https://www.tcom242242.net/entry/ai-2/%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92/%E6%B7%B1%E5%B1%A4%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92/%E3%80%90%E6%B7%B1%E5%B1%A4%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92%E3%80%91deep_q_network_%E3%82%92tensorflow%E3%81%A7%E5%AE%9F%E8%A3%85/
def to_osero():

    IN = [0] * (BOARD_SIZE * BOARD_SIZE) # 状態
    OUT = [0] * (BOARD_SIZE * BOARD_SIZE) # Q値
    ACT = np.arange(BOARD_SIZE * BOARD_SIZE) # 行動：[0,1,2,3,...]

    # クラスのインスタンス化
    Target_network = net()
    board = Board(BOARD_SIZE) 

    # 重みのロード
    w2 = np.load("weight/w2_6_1027.npy")
    w3 = np.load("weight/w3_6_1027.npy")

    # 記録の可視化
    black_win = 0
    black_win_interval = 0
    black_win_rate = []
    black_win_rate_all = []
    white_win = 0
    draw = 0

    # テスト数
    NB_EPISODE = 1000

    flag = 0

    for episode in range(0, NB_EPISODE):
        while True: # 1 game play：board.Turnsでplayerを判定
            if board.Turns % 2 == 0: # DQN
                # 状態を変換
                state = trans(board.RawBoard)
    
                # ネットワークに打ち込む
                buf = Target_network.ForwardPropagation(state, w2, w3)

                # 最大Q値の行動選択
                action = ACT[np.argmax(buf['z3'])]
                    
                # 一度ワンクッションしてる（後で直したい）
                action_board = (IN_ALPHABET[int(action % BOARD_SIZE)],IN_NUMBER[int(action // BOARD_SIZE)])

                 # 最大Q値の行動が無効手であった時、ランダム行動
                if flag == 1:
                    action_board = (IN_ALPHABET[random.randint(0,7)],IN_NUMBER[random.randint(0,7)])
                    flag = 0

                # 入力手をチェック（基本的に範囲内にある）
                if board.checkIN(action_board):
                    x = IN_ALPHABET.index(action_board[0]) + 1
                    y = IN_NUMBER.index(action_board[1]) + 1

                # 手を打つ
                if not board.move(x, y):
                    flag = 1
                    continue

            else: # random

                action_board = (IN_ALPHABET[random.randint(0,7)],IN_NUMBER[random.randint(0,7)])

                if board.checkIN(action_board):
                    x = IN_ALPHABET.index(action_board[0]) + 1
                    y = IN_NUMBER.index(action_board[1]) + 1

                if not board.move(x, y):
                    continue

            # board.display()

            # 勝敗の判定 
            if board.isGameOver():
                count_black = np.count_nonzero(board.RawBoard[:, :] == BLACK)
                count_white = np.count_nonzero(board.RawBoard[:, :] == WHITE)
                dif = count_black - count_white
                if dif > 0:
                    black_win += 1
                    black_win_interval += 1
                elif dif < 0:
                    white_win += 1
                elif dif == 0:
                    draw += 1
                break
            
            # パス判定
            if not board.MovablePos[:, :].any():
                board.CurrentColor = - board.CurrentColor
                board.initMovable()
                continue
        
        #ボードの初期化
        board.__init__(BOARD_SIZE)
        
        episode_interval = 20 # 可視化間隔
        if episode % episode_interval == 0 and episode != 0:

            print('Interval Learning -- episodes:' + str(episode))
            print('black_win_interval_rate：' + str(black_win_interval / episode_interval))

            black_win_rate.append(black_win_interval / episode_interval)
            black_win_rate_all.append(black_win / episode)
            black_win_interval = 0
       

    #------------------------結果の可視化------------------------
    print('Black WIN：' + str(black_win))
    print('White WIN：' + str(white_win))
    print('draw：' + str(draw))
    print(black_win_rate)
    x = range(len(black_win_rate))
    plt.plot(x,black_win_rate,color = 'blue', marker = 'v')
    plt.plot(x,black_win_rate_all,color = 'orange', marker = 'o')
    plt.show()
    #------------------------結果の可視化------------------------

if __name__ == '__main__':

    to_osero()