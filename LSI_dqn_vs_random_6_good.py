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
    Q_network = net()
    board = Board(BOARD_SIZE) 
    memory = ExperienceMemory(400)

    # 重みの初期化　学習時のランダム初期値で結果が異なる場合がある。
    n1 = len(IN) # 入力の要素数
    n3 = len(OUT) # 入力の要素数
    n2 = BOARD_SIZE * BOARD_SIZE # 中間層のユニット数
    w2 = np.random.normal(0,1,(n2,n1))
    w2 = np.insert(w2,0,0,axis=1)
    w3 = np.random.normal(0,1,(n3,n2))
    w3 = np.insert(w3,0,0,axis=1)

    # 記録の可視化
    error = []
    black_win = 0
    black_win_interval = 0
    black_win_rate = []
    white_win = 0
    draw = 0

    # エピソード数
    NB_EPISODE = 800

    # εグリーディー戦略
    epsilon_start = 0.9
    epsilon_end = 0.05
    # psilon_decay = NB_EPISODE / 100 # ネット見た感じ、エピソード数 / 100 くらいがいい値な気がする。
    epsilon_decay = 20 # 大きいほど、ランダム行動の割合が多くなる。"エピソード数との兼ね合いで向上が可能かも"
    
    for episode in range(0, NB_EPISODE):
        while True: # 1 game play：board.Turnsで打ちてを判定 4×4だと後攻有利なので、先行DQN
            if board.Turns % 2 == 0: # DQN
                # 状態を観測する
                state = trans(board.RawBoard)
    
                # 状態をTarget NetworkQ(s′,a|θ−)に入力：重みは経験学習まで一定
                buf = Target_network.ForwardPropagation(state, w2, w3)

                # Target Networkから出力されたQ値を元に,ε-greedy選択法で行動選択
                epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(- episode / epsilon_decay)
                if np.random.uniform() < epsilon:
                    # ランダム行動
                    action = np.random.randint(0, len(ACT))
                else:   
                    # 最大Q値の行動選択
                    action = ACT[np.argmax(buf['z3'])]
                
                # 一度ワンクッションしてる（後で直したい）
                action_board = (IN_ALPHABET[int(action % BOARD_SIZE)],IN_NUMBER[int(action // BOARD_SIZE)])

                # 入力手をチェック（基本的に範囲内にある）
                if board.checkIN(action_board):
                    x = IN_ALPHABET.index(action_board[0]) + 1
                    y = IN_NUMBER.index(action_board[1]) + 1

                # 手を打つ（相手が打ってから、経験を保存する。）
                if not board.move(x, y):
                    continue
                else:
                    # 有効手が角であった時にflagを立て、相手行動時に+報酬を与える。
                    Corner_flag = 0
                    if x == 1 and y == 1:
                        Corner_flag = 1
                    if x == 1 and y == BOARD_SIZE:
                        Corner_flag = 1
                    if x == BOARD_SIZE and y == 1:
                        Corner_flag = 1
                    if x == BOARD_SIZE and y == BOARD_SIZE:
                        Corner_flag = 1
                    # 有効手が悪手:Xであった時に、マイナス報酬
                    Bad_X_hand_flag = 0
                    if x == 2 and y == 2:
                         Bad_X_hand_flag = 1
                    if x == 2 and y == BOARD_SIZE - 1:
                         Bad_X_hand_flag = 1
                    if x == BOARD_SIZE - 1 and y == 2:
                         Bad_X_hand_flag = 1
                    if x == BOARD_SIZE - 1 and y == BOARD_SIZE - 1:
                         Bad_X_hand_flag = 1
                    # 有効手が悪手:Cであった時に、マイナス報酬
                    Bad_C_hand_flag = 0
                    if x == 1 and y == 2:
                        Bad_C_hand_flag = 1
                    if x == 2 and y == 1:
                        Bad_C_hand_flag = 1
                    if x == 1 and y == BOARD_SIZE - 1:
                        Bad_C_hand_flag = 1
                    if x == 2 and y == BOARD_SIZE:
                        Bad_C_hand_flag = 1
                    if x == BOARD_SIZE - 1 and y == 1:
                        Bad_C_hand_flag = 1
                    if x == BOARD_SIZE and y == 2:
                        Bad_C_hand_flag = 1
                    if x == BOARD_SIZE and y == BOARD_SIZE - 1:
                        Bad_C_hand_flag = 1
                    if x == BOARD_SIZE - 1 and y == BOARD_SIZE:
                        Bad_C_hand_flag = 1

            else: # random

                # randomに行動の選択
                action_board = (IN_ALPHABET[random.randint(0,7)],IN_NUMBER[random.randint(0,7)])

                if board.checkIN(action_board):
                    x = IN_ALPHABET.index(action_board[0]) + 1
                    y = IN_NUMBER.index(action_board[1]) + 1

                if not board.move(x, y):
                    continue
                else:
                    # 状態 s′ と報酬rの観測
                    state_next = trans(board.RawBoard)
                    if Corner_flag == 1:# 角判定
                        reward = 0.2
                        # print("DQN good corner")
                    elif Bad_X_hand_flag == 1:
                        reward = -0.1
                    elif Bad_C_hand_flag == 1:
                        reward = -0.1
                    else:
                        reward = 0
                    if board.Turns > 0:# state,actionがないので、random初手は経験の保存無し
                        memory.push(state,action,state_next,reward)

            # board.display()

            # 勝敗の判定 
            if board.isGameOver():
                count_black = np.count_nonzero(board.RawBoard[:, :] == BLACK)
                count_white = np.count_nonzero(board.RawBoard[:, :] == WHITE)
                state_next = trans(board.RawBoard)
                dif = count_black - count_white
                if dif > 0:
                    black_win += 1
                    black_win_interval += 1
                    reward = 1
                elif dif < 0:
                    white_win += 1
                    reward = -1
                elif dif == 0:
                    draw += 1
                    reward = 0
                state_next = trans(board.RawBoard)
                memory.push(state,action,state_next,reward)
                break
            
            # パス判定
            if not board.MovablePos[:, :].any():
                board.CurrentColor = - board.CurrentColor
                board.initMovable()
                continue
        
        #ボードの初期化
        board.__init__(BOARD_SIZE)
        
        episode_interval = 80 #  80くらいがいい。あんまりエピソード数に依存しない、同期頻度：短いと学習が不安定化し、長いと学習が進みにくくなる。ハイパーパラメータの１つ
        # 5.（定期動作）Experience Bufferから任意の経験を取り出し、Q Networkをミニバッチ学習(Experience Replay)
        if episode % episode_interval == 0 and episode != 0:
            w2_init = copy.deepcopy(w2) # Target_network用に重みを固定
            w3_init = copy.deepcopy(w3) # Target_network用に重みを固定
            # 経験をランダムサンプリング
            memory_num = episode_interval * 4 # 間隔数だけ経験を抜き取る
            data = memory.sample(memory_num)
            # Q Networkの学習実行：dataからミニバッチ法でやりたい
            Epoch_Q = 400 # エポック数
            Bach_Size_Q = episode_interval
            for _ in range(0,Epoch_Q):
                Random_index = random.sample(range(memory_num), k=memory_num)
                Batc_count = 0
                for k in range(0, memory_num // Bach_Size_Q):
                    Batc_count = Bach_Size_Q * k
                    for i in range(Batc_count,Bach_Size_Q + Batc_count):
                        buf_data = data[Random_index[i]]
                        #buf_data = data[i]

                        # 遷移状態におけるTarget_network出力のmaxQ値を取得
                        buf = Target_network.ForwardPropagation(buf_data.next_state, w2_init, w3_init)

                        # maxQ値を今回の行動値にセットし、それ以外を0でマスク処理
                        # 必要か分かんない
                        max_Q = buf['z3'][0]
                        Q_nextstae_max = copy.deepcopy(buf['z3'])
                        for ii in range(0,BOARD_SIZE * BOARD_SIZE):
                            Q_nextstae_max[ii] = 0
                            if max_Q < buf['z3'][ii]:
                                max_Q = buf['z3'][ii]
                        Q_nextstae_max[buf_data.action] = buf_data.reward + 0.99 * max_Q
                        y = copy.deepcopy(Q_nextstae_max)

                        Y_train = np.array(y)

                        # 順伝播計算をして、fに係数を辞書る。
                        f = Q_network.ForwardPropagation(buf_data.state,w2,w3)
                        
                        # 誤差逆伝播法により、bに勾配を辞書る
                        b = Q_network.BackPropagation(Y_train,w2,w3,f['z1'],f['z2'],f['z3'],f['u2'])

                        # 勾配に基づいて重みを更新
                        epsilon = 0.1 # 時によっては、ε-greedy行動選択と同じ値を使ってた。バッチサイズを合わせて調整。0.1くらいがいいかも
                        w2 = copy.deepcopy(w2 - epsilon*b['dw2'] / Bach_Size_Q)
                        w3 = copy.deepcopy(w3 - epsilon*b['dw3'] / Bach_Size_Q)
                        d3 = b['d3']

                        error.append(max(max(abs(d3), key=max)))
                        #print(w['w2'])


            print('Interval Learning -- episodes:' + str(episode))
            print('black_win_interval_rate：' + str(black_win_interval / episode_interval))
            print('loss max：' + str(max(max(abs(d3), key=max))))

            # 経験の初期化
            memory.__init__(400)
            black_win_rate.append(black_win_interval / episode_interval)

            # 学習が上手くいかない時に、重みを初期化する。
            if black_win_interval / episode_interval < 0.03 and episode >= 100:
                    w2 = np.random.normal(0,1,(n2,n1))
                    w2 = np.insert(w2,0,0,axis=1)
                    w3 = np.random.normal(0,1,(n3,n2))
                    w3 = np.insert(w3,0,0,axis=1)
                    print("----------重みを初期化しました。----------")

            black_win_interval = 0

       

    #------------------------結果の可視化------------------------
    print('Black WIN：' + str(black_win))
    print('White WIN：' + str(white_win))
    print('draw：' + str(draw))
    print(black_win_rate)
    x = range(len(black_win_rate))
    plt.plot(x,black_win_rate)
    # 傾きが変化していれば、学習していると思う。報酬の与え方で、上か下か
    plt.show()
    x = range(len(error))
    plt.plot(x,error)
    # plt.savefig("black rate.png")
    # 経験を学習する毎における、学習時の誤差の最大値を表示
    plt.show()
    #------------------------結果の可視化------------------------

    # 重みの保存
    np.save("lsi_reverse/weight/w2_6",w2)
    np.save("lsi_reverse/weight/w3_6",w3)

if __name__ == '__main__':

    to_osero()