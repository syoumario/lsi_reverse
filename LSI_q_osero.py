import numpy as np
import random
import matplotlib.pyplot as plt
import copy
from Board.board import Board

# マスの状態
EMPTY = 0 # 空きマス
WHITE = -1 # 白石
BLACK = 1 # 黒石
WALL = 2 # 壁
 
# ボードのサイズ
BOARD_SIZE = 4

# 方向(2進数)
NONE = 0
LEFT = 2**0 # =1
UPPER_LEFT = 2**1 # =2 
UPPER = 2**2 # =4
UPPER_RIGHT = 2**3 # =8
RIGHT = 2**4 # =16
LOWER_RIGHT = 2**5 # =32
LOWER = 2**6 # =64
LOWER_LEFT = 2**7 # =128
 
# 手の表現
IN_ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
IN_NUMBER = ['1', '2', '3', '4', '5', '6', '7', '8']
 
# 手数の上限
MAX_TURNS = 60

#Q値：辞書管理(str型環境 : 行動の集合)
class QLearningAgent:
    """
        Q学習 エージェント
    """
    # インスタンスの初期設定
    def __init__(self,alpha=.2,epsilon=.1,gamma=.99,actions=None,observation=None):
        self.alpha = alpha                      #学習率：TD誤差をどれだけ反映させるか
        self.gamma = gamma                      #割引率：遷移先の最大Q値をどれだけ利用するか
        self.epsilon = epsilon                  #探索率：ランダムに行動する確立
        self.reward_history = []                #報酬履歴：
        self.actions = actions                  #行動の集合：行動の選択肢
        self.state = str(observation)           #環境：エージェントの現在地点(x, y)
        self.ini_state = str(observation)       #初期環境：エージェントのスタート地点(x, y)
        self.previous_state = None              #前環境：エージェントの前地点(x, y)
        self.previous_action = None             #行動の集合：エージェントの前行動(x, y)
        self.q_values = self._init_q_values()   #Q値テーブル：

    def _init_q_values(self):
        """
           Q テーブルの初期化
        """
        q_values = {}#辞書生成
        # np.repeat(a,n)：配列aの各要素をn回繰り返す配列を生成
        # 初期エージェントポジにおいて、行動数の配列を生成：q_values[init_x,init_y] = [0.0 0.0 0.0 0.0]
        q_values[self.state] = np.repeat(0.0, len(self.actions))
        return q_values

    def init_state(self):
        """
            状態の初期化　使ってない
        """
        # copy.deepcop()：新たに生成して、コピーする。
        self.previous_state = copy.deepcopy(self.ini_state)
        self.state = copy.deepcopy(self.ini_state)
        return self.state

    def act(self):
        # ε-greedy選択
        # np.random.uniform():0～1のランダム値
        if np.random.uniform() < self.epsilon:  # random行動
            # Qテーブルにおける現在位置のQ値の範囲内で、ランダムに行動を選択
            action = np.random.randint(0, len(self.q_values[self.state]))
        else:   # greedy 行動
            # np.argmax：配列の最大要素数のインデックスを返す
            # Qテーブルにおける現在環境のQ値を最大にとるインデックス（0~3）を選択
            action = np.argmax(self.q_values[self.state])
        # 次エピソードに向けて、前行動に現在行動をセット
        #self.previous_action = action
        # 行動を返す（0~3）
        return action

    def observe(self, next_state, reward=None):
        """
            次の状態と報酬の観測
        """
        next_state = str(next_state)# エージェントのポジ
        if next_state not in self.q_values:  # 始めて訪れる状態であれば
            # その場所におけるQ値を新たに行動毎に作る
            self.q_values[next_state] = np.repeat(0.0, len(self.actions))
        
        # 次エピソードに向けて、前ポジに現在ポジをセット
        self.previous_state = copy.deepcopy(self.state)
        # 現在位置を更新
        self.state = next_state

        # 報酬を記憶
        if reward is not None:
            # 報酬履歴に入れ込む
            self.reward_history.append(reward)
            # 報酬を元にQ値を更新
            self.learn(reward)

    def learn(self, reward):
        """
            Q値の更新
        """
        q = self.q_values[self.previous_state][self.previous_action]  # Q(s, a)
        max_q = max(self.q_values[self.state])  # max Q(s')
        # Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        self.q_values[self.previous_state][self.previous_action] = q + \
            (self.alpha * (reward + (self.gamma * max_q) - q))

'''
この場合だと、{報酬が与えられる状態まで進み、
与えられた報酬を基に今まで行ってきた『状態と行動のセット』を評価する}
の「今まで行ってきた『状態と行動のセット』を評価する」が未完成で、
勝敗がついて報酬が与えられた時の直前の『状態と行動のセット』しか評価されてないと思うのでだめと思う
今まで行ってきた『状態と行動のセット』も評価する必要
'''
if __name__ == '__main__':
    board = Board()  # オセロ環境
    ini_state = board.RawBoard  # 初期状態：

    # エージェントの初期化
    agent = QLearningAgent(
        alpha= 0.1,
        gamma= 0.90,
        epsilon= 0.1,  # 探索率
        actions= np.arange(BOARD_SIZE * BOARD_SIZE),   # 行動の集合
        observation=ini_state)  # Q学習エージェント
    rewards = []    # 評価用報酬の保存
    
    black_win = 0
    white_win = 0
    draw = 0
    
    # 学習
    NB_EPISODE = 3000
    for episode in range(NB_EPISODE):
        episode_reward = []  # 1エピソードの累積報酬
        
        while True:
            # コンピュータ　== : 黒（先手）,!= 白（後手）
            if board.Turns % 2 != 0:
                IN = (IN_ALPHABET[random.randint(0,7)],IN_NUMBER[random.randint(0,7)])
                # 入力手をチェック
                if board.checkIN(IN):
                    x = IN_ALPHABET.index(IN[0]) + 1
                    y = IN_NUMBER.index(IN[1]) + 1
                else:
                    continue

                # 手を打つ
                if not board.move(x, y):
                    continue
                else:
                    count_black = np.count_nonzero(board.RawBoard[:, :] == BLACK)
                    count_white = np.count_nonzero(board.RawBoard[:, :] == WHITE)
                    if count_black - count_white > 0:
                        #現地点で黒が多かったら
                        agent.observe(board.RawBoard, 0)
                    else:#現地点で白が多かったら
                        agent.observe(board.RawBoard, 0)

                # 盤面の表示
                #board.display()
 
                # 終局判定
                if board.isGameOver():
                    #grid_env.display()
                    break
 
                # パス
                if not board.MovablePos[:, :].any():
                    board.CurrentColor = - board.CurrentColor
                    board.initMovable()
                    print()
                    continue

            # エージェント
            else:
                # 行動選択して、入力手に変換
                action = agent.act() 
                #action_board = (IN_ALPHABET[random.randint(0,7)],IN_NUMBER[random.randint(0,7)])
                action_board = (IN_ALPHABET[int(action % BOARD_SIZE)],IN_NUMBER[int(action // BOARD_SIZE)])
                
                # 入力手をチェック
                if board.checkIN(action_board):
                    x = IN_ALPHABET.index(action_board[0]) + 1
                    y = IN_NUMBER.index(action_board[1]) + 1
                else:
                    print("エラーです。")
                    agent.observe(board.RawBoard, 0)
                    episode_reward.append(0)
                    continue
 
                # 手を打つ
                if not board.move(x, y):
                    # その入力手では打てないため、マイナス報酬
                    agent.observe(board.RawBoard, 0)
                    episode_reward.append(0)
                    continue
                else:
                    ##ここで実際に手が打てた状態
                    agent.previous_action = action #clas:act()内でやらず、有効手の場合に、有効手を更新
                    count_black = np.count_nonzero(board.RawBoard[:, :] == BLACK)
                    count_white = np.count_nonzero(board.RawBoard[:, :] == WHITE)
                    if count_black - count_white > 0:
                        #現地点で黒が多かったら
                        agent.observe(board.RawBoard, 0)
                    else:#現地点で白が多かったら
                        agent.observe(board.RawBoard, 0)

                # 盤面の表示
                #board.display()
 
                # 終局判定
                if board.isGameOver():
                    #board.display()
                    break
 
                # パス
                if not board.MovablePos[:, :].any():
                    board.CurrentColor = - board.CurrentColor
                    board.initMovable()
                    #print('パスしました')
                    print()
                    agent.observe(board.RawBoard, 0)
                    episode_reward.append(0)
                    continue
                
        
        if episode % 10 == 0:
            print(episode)

        ## 各色の数
        count_black = np.count_nonzero(board.RawBoard[:, :] == BLACK)
        count_white = np.count_nonzero(board.RawBoard[:, :] == WHITE)

        ## 勝敗判定
        dif = count_black - count_white
        if dif > 0:#先手（黒）が勝つ
            agent.observe(board.RawBoard, 10)
            episode_reward.append(10)
            black_win += 1
        elif dif < 0:#後手（白）が勝つ
            agent.observe(board.RawBoard, -10)
            episode_reward.append(-10)
            white_win += 1
        elif dif == 0:#引き分け
            agent.observe(board.RawBoard, 0)
            draw += 1
        
        rewards.append(np.sum(episode_reward)) 

        #ボードの初期化
        board.__init__()
        

    # 結果のプロット
    plt.plot(np.arange(NB_EPISODE), rewards)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.savefig("result.jpg")
    print(rewards)
    #print(agent.q_values)
    print('black_win：' + str(black_win))
    print('white_win：' + str(white_win))
    print('draw：' + str(draw))
    #print(agent.q_values)

    plt.show()