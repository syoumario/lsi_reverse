import numpy as np
import random
import matplotlib.pyplot as plt
import copy

# マスの状態
EMPTY = 0 # 空きマス
WHITE = -1 # 白石
BLACK = 1 # 黒石
WALL = 2 # 壁
 
# ボードのサイズ
BOARD_SIZE = 6
 
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

# 環境：self.RawBoard = [[BOARD_SIZE + 2] * (BOARD_SIZE + 2)]
# 行動：x,y = x:1 ~ BOARD_SIZE - 1, y:1 ~ BOARD_SIZE - 1
# 手番数：self.Turns = 0 → 先手黒スタート
class Board:

    def __init__(self):
        """
        ボードの表現
        """ 
 
        # 全マスを空きマスに設定
        self.RawBoard = np.zeros((BOARD_SIZE + 2, BOARD_SIZE + 2), dtype=int)
 
        # 壁の設定
        self.RawBoard[0, :] = WALL
        self.RawBoard[:, 0] = WALL
        self.RawBoard[BOARD_SIZE + 1, :] = WALL
        self.RawBoard[:, BOARD_SIZE + 1] = WALL
 
        # 初期配置
        self.RawBoard[BOARD_SIZE //2 , BOARD_SIZE //2] = WHITE
        self.RawBoard[BOARD_SIZE //2 + 1, BOARD_SIZE //2 + 1] = WHITE
        self.RawBoard[BOARD_SIZE //2, BOARD_SIZE //2 + 1] = BLACK
        self.RawBoard[BOARD_SIZE //2 + 1, BOARD_SIZE //2] = BLACK
 
        # 手番
        self.Turns = 0
 
        # 現在の手番の色
        self.CurrentColor = BLACK
 
        # 置ける場所と石が返る方向
        self.MovablePos = np.zeros((BOARD_SIZE + 2, BOARD_SIZE + 2), dtype=int)
        self.MovableDir = np.zeros((BOARD_SIZE + 2, BOARD_SIZE + 2), dtype=int)
 
        # MovablePosとMovableDirを初期化
        self.initMovable()
 
    def checkMobility(self, x, y, color):
        """
        どの方向に石が裏返るかをチェック
        """
        # 注目しているマスの裏返せる方向の情報が入る
        dir = 0
 
        # 既に石がある場合はダメ
        if(self.RawBoard[x, y] != EMPTY):
            return dir
 
        ## 左
        if(self.RawBoard[x - 1, y] == - color): # 直上に相手の石があるか
            
            x_tmp = x - 2
            y_tmp = y
 
            # 相手の石が続いているだけループ
            while self.RawBoard[x_tmp, y_tmp] == - color:
                x_tmp -= 1
            
            # 相手の石を挟んで自分の石があればdirを更新
            if self.RawBoard[x_tmp, y_tmp] == color:
                dir = dir | LEFT
 
        ## 左上
        if(self.RawBoard[x - 1, y - 1] == - color): # 直上に相手の石があるか
            
            x_tmp = x - 2
            y_tmp = y - 2
            
            # 相手の石が続いているだけループ
            while self.RawBoard[x_tmp, y_tmp] == - color:
                x_tmp -= 1
                y_tmp -= 1
            
            # 相手の石を挟んで自分の石があればdirを更新
            if self.RawBoard[x_tmp, y_tmp] == color:
                dir = dir | UPPER_LEFT
 
        ## 上
        if(self.RawBoard[x, y - 1] == - color): # 直上に相手の石があるか
            
            x_tmp = x
            y_tmp = y - 2
            
            # 相手の石が続いているだけループ
            while self.RawBoard[x_tmp, y_tmp] == - color:
                y_tmp -= 1
            
            # 相手の石を挟んで自分の石があればdirを更新
            if self.RawBoard[x_tmp, y_tmp] == color:
                dir = dir | UPPER
 
        ## 右上
        if(self.RawBoard[x + 1, y - 1] == - color): # 直上に相手の石があるか
            
            x_tmp = x + 2
            y_tmp = y - 2
            
            # 相手の石が続いているだけループ
            while self.RawBoard[x_tmp, y_tmp] == - color:
                x_tmp += 1
                y_tmp -= 1
            
            # 相手の石を挟んで自分の石があればdirを更新
            if self.RawBoard[x_tmp, y_tmp] == color:
                dir = dir | UPPER_RIGHT
 
        ## 右
        if(self.RawBoard[x + 1, y] == - color): # 直上に相手の石があるか
 
            x_tmp = x + 2
            y_tmp = y
            
            # 相手の石が続いているだけループ
            while self.RawBoard[x_tmp, y_tmp] == - color:
                x_tmp += 1
            
            # 相手の石を挟んで自分の石があればdirを更新
            if self.RawBoard[x_tmp, y_tmp] == color:
                dir = dir | RIGHT
 
        ## 右下
        if(self.RawBoard[x + 1, y + 1] == - color): # 直上に相手の石があるか
            
            x_tmp = x + 2
            y_tmp = y + 2
            
            # 相手の石が続いているだけループ
            while self.RawBoard[x_tmp, y_tmp] == - color:
                x_tmp += 1
                y_tmp += 1
            
            # 相手の石を挟んで自分の石があればdirを更新
            if self.RawBoard[x_tmp, y_tmp] == color:
                dir = dir | LOWER_RIGHT
 
        ## 下
        if(self.RawBoard[x, y + 1] == - color): # 直上に相手の石があるか
            
            x_tmp = x
            y_tmp = y + 2
            
            # 相手の石が続いているだけループ
            while self.RawBoard[x_tmp, y_tmp] == - color:
                y_tmp += 1
            
            # 相手の石を挟んで自分の石があればdirを更新
            if self.RawBoard[x_tmp, y_tmp] == color:
                dir = dir | LOWER
 
        ## 左下
        if(self.RawBoard[x - 1, y + 1] == - color): # 直上に相手の石があるか
            
            x_tmp = x - 2
            y_tmp = y + 2
            
            # 相手の石が続いているだけループ
            while self.RawBoard[x_tmp, y_tmp] == - color:
                x_tmp -= 1
                y_tmp += 1
            
            # 相手の石を挟んで自分の石があればdirを更新
            if self.RawBoard[x_tmp, y_tmp] == color:
                dir = dir | LOWER_LEFT
 
        return dir
 
    def flipDiscs(self, x, y):
        """
        石を置くことによる盤面の変化をボードに反映
        """
 
        # 石を置く
        self.RawBoard[x, y] = self.CurrentColor
 
        # 石を裏返す
        # MovableDirの(y, x)座標をdirに代入
        dir = self.MovableDir[x, y]
 
        ## 左
        if dir & LEFT: # AND演算子
 
            x_tmp = x - 1
 
            # 相手の石がある限りループが回る
            while self.RawBoard[x_tmp, y] == - self.CurrentColor:
 
                # 相手の石があるマスを自分の石の色に塗り替えている
                self.RawBoard[x_tmp, y] = self.CurrentColor
 
                # さらに1マス左に進めてループを回す
                x_tmp -= 1
 
        ## 左上
        if dir & UPPER_LEFT: # AND演算子
 
            x_tmp = x - 1
            y_tmp = y - 1
 
            # 相手の石がある限りループが回る
            while self.RawBoard[x_tmp, y_tmp] == - self.CurrentColor:
 
                # 相手の石があるマスを自分の石の色に塗り替えている
                self.RawBoard[x_tmp, y_tmp] = self.CurrentColor
                
                # さらに1マス左上に進めてループを回す
                x_tmp -= 1
                y_tmp -= 1
 
        ## 上
        if dir & UPPER: # AND演算子
 
            y_tmp = y - 1
 
            # 相手の石がある限りループが回る
            while self.RawBoard[x, y_tmp] == - self.CurrentColor:
 
                # 相手の石があるマスを自分の石の色に塗り替えている
                self.RawBoard[x, y_tmp] = self.CurrentColor
 
                # さらに1マス上に進めてループを回す
                y_tmp -= 1
 
        ## 右上
        if dir & UPPER_RIGHT: # AND演算子
 
            x_tmp = x + 1
            y_tmp = y - 1
 
            # 相手の石がある限りループが回る
            while self.RawBoard[x_tmp, y_tmp] == - self.CurrentColor:
 
                # 相手の石があるマスを自分の石の色に塗り替えている
                self.RawBoard[x_tmp, y_tmp] = self.CurrentColor
 
                # さらに1マス右上に進めてループを回す
                x_tmp += 1
                y_tmp -= 1
 
        ## 右
        if dir & RIGHT: # AND演算子
 
            x_tmp = x + 1
 
            # 相手の石がある限りループが回る
            while self.RawBoard[x_tmp, y] == - self.CurrentColor:
 
                # 相手の石があるマスを自分の石の色に塗り替えている
                self.RawBoard[x_tmp, y] = self.CurrentColor
                
                # さらに1マス右に進めてループを回す
                x_tmp += 1
 
        ## 右下
        if dir & LOWER_RIGHT: # AND演算子
 
            x_tmp = x + 1
            y_tmp = y + 1
 
            # 相手の石がある限りループが回る
            while self.RawBoard[x_tmp, y_tmp] == - self.CurrentColor:
 
                # 相手の石があるマスを自分の石の色に塗り替えている
                self.RawBoard[x_tmp, y_tmp] = self.CurrentColor
 
                # さらに1マス右下に進めてループを回す
                x_tmp += 1
                y_tmp += 1
 
        ## 下
        if dir & LOWER: # AND演算子
 
            y_tmp = y + 1
 
            # 相手の石がある限りループが回る
            while self.RawBoard[x, y_tmp] == - self.CurrentColor:
 
                # 相手の石があるマスを自分の石の色に塗り替えている
                self.RawBoard[x, y_tmp] = self.CurrentColor
 
                # さらに1マス下に進めてループを回す
                y_tmp += 1
 
        ## 左下
        if dir & LOWER_LEFT: # AND演算子
 
            x_tmp = x - 1
            y_tmp = y + 1
 
            # 相手の石がある限りループが回る
            while self.RawBoard[x_tmp, y_tmp] == - self.CurrentColor:
                
                # 相手の石があるマスを自分の石の色に塗り替えている
                self.RawBoard[x_tmp, y_tmp] = self.CurrentColor
 
                # さらに1マス左下に進めてループを回す
                x_tmp -= 1
                y_tmp += 1
        
    def move(self, x, y):
        """
        石を置く
        """
 
        # 置く位置が正しいかどうかをチェック
        if x < 1 or BOARD_SIZE < x:
            return False
        if y < 1 or BOARD_SIZE < y:
            return False
        if self.MovablePos[x, y] == 0:
            return False
 
        # 石を裏返す
        self.flipDiscs(x, y)
 
        # 手番を進める
        self.Turns += 1
 
        # 手番を交代する
        self.CurrentColor = - self.CurrentColor
        
        # MovablePosとMovableDirの更新
        self.initMovable()
 
        return True
 
    def initMovable(self):
        """
        MovablePosとMovableDirの更新
        """
        # MovablePosの初期化（すべてFalseにする）
        self.MovablePos[:, :] = False
 
        # すべてのマス（壁を除く）に対してループ
        for x in range(1, BOARD_SIZE + 1):
            for y in range(1, BOARD_SIZE + 1):
 
                # checkMobility関数の実行
                dir = self.checkMobility(x, y, self.CurrentColor)
 
                # 各マスのMovableDirにそれぞれのdirを代入
                self.MovableDir[x, y] = dir
 
                # dirが0でないならMovablePosにTrueを代入
                if dir != 0:
                    self.MovablePos[x, y] = True
                
    def isGameOver(self):
        """
        終局判定
        """
        # 60手に達していたらゲーム終了
        if self.Turns >= MAX_TURNS:
            return True
 
        # (現在の手番)打てる手がある場合はゲームを終了しない
        if self.MovablePos[:, :].any():
            return False
 
        # (相手の手番)打てる手がある場合はゲームを終了しない
        for x in range(1, BOARD_SIZE + 1):
            for y in range(1, BOARD_SIZE + 1):
 
                # 置ける場所が1つでもある場合はゲーム終了ではない
                if self.checkMobility(x, y, - self.CurrentColor) != 0:
                    return False
 
        # ここまでたどり着いたらゲームは終わっている
        return True
 
    def skip(self):
        """
        パスの判定
        """
 
        # すべての要素が0のときだけパス(1つでも0以外があるとFalse)
        if any(MovablePos[:, :]):
            return False
 
        # ゲームが終了しているときはパスできない
        if isGameOver():
            return False
 
        # ここまで来たらパスなので手番を変える
        self.CurrentColor = - self.CurrentColor
 
        # MovablePosとMovableDirの更新
        self.initMovable()
 
        return True

    def display(self):
        """
        オセロ盤面の表示
        """
        # 横軸
        print(' abcdefgh')
        # 縦軸方向へのマスのループ
        for y in range(1, BOARD_SIZE + 1):
 
            # 縦軸
            print(y, end="")
            # 横軸方向へのマスのループ
            for x in range(1, BOARD_SIZE + 1):
 
                # マスの種類(数値)をgridに代入
                grid = self.RawBoard[x, y]
 
                # マスの種類によって表示を変化
                if grid == EMPTY: # 空きマス
                    print('□', end="")
                elif grid == WHITE: # 白石
                    print('◌', end="")
                elif grid == BLACK: # 黒石
                    print('●', end="")
 
            # 最後に改行
            print()
 
    def checkIN(self, IN):
        """
        入力された手の形式をチェック
        """
        # INが空でないかをチェック
        if not IN:
            return False
 
        # INの1文字目と2文字目がそれぞれa~h,1~8の範囲内であるかをチェック
        if IN[0] in IN_ALPHABET:
            if IN[1] in IN_NUMBER:
                return True
 
        return False

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
        self.previous_action = action
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

if __name__ == '__main__':
    board = Board()  # オセロ環境
    ini_state = board.RawBoard  # 初期状態：

    # エージェントの初期化
    agent = QLearningAgent(
        alpha= 0.1,
        gamma= 0.90,
        epsilon= 0.9,  # 探索率
        actions= np.arange(BOARD_SIZE * BOARD_SIZE),   # 行動の集合
        observation=ini_state)  # Q学習エージェント
    rewards = []    # 評価用報酬の保存
    
    cpu_win = 0
    agent_win = 0
    draw = 0
    
    # 学習
    NB_EPISODE = 200
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
                action_board = (IN_ALPHABET[int(action % BOARD_SIZE)],IN_NUMBER[int(action // BOARD_SIZE)])
                
                # 入力手をチェック
                if board.checkIN(action_board):
                    x = IN_ALPHABET.index(action_board[0]) + 1
                    y = IN_NUMBER.index(action_board[1]) + 1
                else:
                    print("エラーです。")
                    agent.observe(board.RawBoard, -5)
                    episode_reward.append(-1)
                    continue
 
                # 手を打つ
                if not board.move(x, y):
                    # その入力手では打てないため、マイナス報酬
                    agent.observe(board.RawBoard, 0)
                    episode_reward.append(-1)
                    continue
                else:
                    count_black = np.count_nonzero(board.RawBoard[:, :] == BLACK)
                    count_white = np.count_nonzero(board.RawBoard[:, :] == WHITE)
                    if count_black - count_white > 0:
                        #現地点で黒が多かったら
                        agent.observe(board.RawBoard, 10)
                    else:#現地点で白が多かったら
                        agent.observe(board.RawBoard, -10)

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
                    episode_reward.append(-1)
                    continue
                
        
        if episode % 10 == 0:
            print(episode)

        ## 各色の数
        count_black = np.count_nonzero(board.RawBoard[:, :] == BLACK)
        count_white = np.count_nonzero(board.RawBoard[:, :] == WHITE)

        ## 勝敗判定
        dif = count_black - count_white
        if dif > 0:#先手（黒）が勝つ
            #agent.observe(board.RawBoard, -1000)
            episode_reward.append(100)
            cpu_win += 1
        elif dif < 0:#後手（白）が勝つ
            #agent.observe(board.RawBoard, 1000)
            episode_reward.append(-100)
            agent_win += 1
        elif dif == 0:#引き分け
            #agent.observe(board.RawBoard, 0)
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
    print('black_win：' + str(cpu_win))
    print('white_win：' + str(agent_win))
    print('draw：' + str(draw))
    #print(agent.q_values)