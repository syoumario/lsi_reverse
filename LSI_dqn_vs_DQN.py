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
    Target_network_a = net()
    Q_network_a = net()
    Target_network_b = net()
    Q_network_b = net()
    board = Board(BOARD_SIZE) 
    memory_a = ExperienceMemory(400)
    memory_b = ExperienceMemory(400)

    # 重みの初期化　学習時のランダム初期値で結果が異なる場合がある。
    n1 = len(IN) 
    n3 = len(OUT) 
    n2 = BOARD_SIZE * BOARD_SIZE 
    w2_a = np.random.normal(0,1,(n2,n1))
    w2_a = np.insert(w2_a,0,0,axis=1)
    w3_a = np.random.normal(0,1,(n3,n2))
    w3_a = np.insert(w3_a,0,0,axis=1)
    w2_b = np.random.normal(0,1,(n2,n1))
    w2_b = np.insert(w2_b,0,0,axis=1)
    w3_b = np.random.normal(0,1,(n3,n2))
    w3_b = np.insert(w3_b,0,0,axis=1)

    # 記録の可視化
    error_a = []
    error_b = []
    black_win = 0
    black_win_interval = 0
    black_win_rate = []
    white_win = 0
    draw = 0

    # エピソード数
    NB_EPISODE = 800
    
    Corner_flag_a = 0                
    Bad_X_hand_flag_a = 0
    Bad_C_hand_flag_a = 0
    Corner_flag_b = 0                
    Bad_X_hand_flag_b = 0
    Bad_C_hand_flag_b = 0

    for episode in range(0, NB_EPISODE):
        if episode % 20 == 0:
            print('episode：' + str(episode))
        while True: # 1 game play：board.Turnsで打ちてを判定 4×4だと後攻有利なので、先行DQN
            if board.Turns % 2 == 0: # DQN_a
                # 状態を観測する
                state_a = trans(board.RawBoard)
    
                # 状態をTarget NetworkQ(s′,a|θ−)に入力：重みは経験学習まで一定
                buf_a = Target_network_a.ForwardPropagation(state_a, w2_a, w3_a)

                # Target Networkから出力されたQ値を元に,ε-greedy選択法で行動選択
                epsilon = 0.2
                if np.random.uniform() < epsilon:
                    # ランダム行動
                    action_a = np.random.randint(0, len(ACT))
                else:   
                    # 最大Q値の行動選択
                    action_a = ACT[np.argmax(buf_a['z3'])]
                
                # 一度ワンクッションしてる（後で直したい）
                action_board = (IN_ALPHABET[int(action_a % BOARD_SIZE)],IN_NUMBER[int(action_a // BOARD_SIZE)])

                # 入力手をチェック（基本的に範囲内にある）
                if board.checkIN(action_board):
                    x = IN_ALPHABET.index(action_board[0]) + 1
                    y = IN_NUMBER.index(action_board[1]) + 1

                # 手を打つ（相手が打ってから、経験を保存する。）
                if not board.move(x, y):
                    continue
                else:
                    # 状態 s′ と報酬rの観測
                    state_next_b = trans(board.RawBoard)
                    if Corner_flag_b == 1:
                        reward_b = 0.2
                    elif Bad_X_hand_flag_b == 1:
                        reward_b = -0.1
                    elif Bad_C_hand_flag_b == 1:
                        reward_b = -0.1
                    else:
                        reward_b = 0
                    if board.Turns > 1:# state,actionがないので、random初手は経験の保存無し
                        memory_b.push(state_b,action_b,state_next_b,reward_b)

                    # 有効手が角であった時にflagを立て、相手行動時に+報酬を与える。
                    Corner_flag_a = 0
                    if x == 1 and y == 1:
                        Corner_flag_a = 1
                    if x == 1 and y == BOARD_SIZE:
                        Corner_flag_a = 1
                    if x == BOARD_SIZE and y == 1:
                        Corner_flag_a = 1
                    if x == BOARD_SIZE and y == BOARD_SIZE:
                        Corner_flag_a = 1
                    # 有効手が悪手:Xであった時に、マイナス報酬
                    Bad_X_hand_flag_a = 0
                    if x == 2 and y == 2:
                         Bad_X_hand_flag_a = 1
                    if x == 2 and y == BOARD_SIZE - 1:
                         Bad_X_hand_flag_a = 1
                    if x == BOARD_SIZE - 1 and y == 2:
                         Bad_X_hand_flag_a = 1
                    if x == BOARD_SIZE - 1 and y == BOARD_SIZE - 1:
                         Bad_X_hand_flag_a = 1
                    # 有効手が悪手:Cであった時に、マイナス報酬
                    Bad_C_hand_flag_a = 0
                    if x == 1 and y == 2:
                        Bad_C_hand_flag_a = 1
                    if x == 2 and y == 1:
                        Bad_C_hand_flag_a = 1
                    if x == 1 and y == BOARD_SIZE - 1:
                        Bad_C_hand_flag_a = 1
                    if x == 2 and y == BOARD_SIZE:
                        Bad_C_hand_flag_a = 1
                    if x == BOARD_SIZE - 1 and y == 1:
                        Bad_C_hand_flag_a = 1
                    if x == BOARD_SIZE and y == 2:
                        Bad_C_hand_flag_a = 1
                    if x == BOARD_SIZE and y == BOARD_SIZE - 1:
                        Bad_C_hand_flag_a = 1
                    if x == BOARD_SIZE - 1 and y == BOARD_SIZE:
                        Bad_C_hand_flag_a = 1

            else: # DQN_b
                # 状態を観測する
                state_b = trans(board.RawBoard)
    
                # 状態をTarget NetworkQ(s′,a|θ−)に入力：重みは経験学習まで一定
                buf_b = Target_network_b.ForwardPropagation(state_b, w2_b, w3_b)

                # Target Networkから出力されたQ値を元に,ε-greedy選択法で行動選択
                epsilon = 0.2
                if np.random.uniform() < epsilon:
                    # ランダム行動
                    action_b = np.random.randint(0, len(ACT))
                else:   
                    # 最大Q値の行動選択
                    action_b = ACT[np.argmax(buf_b['z3'])]
                
                # 一度ワンクッションしてる（後で直したい）
                action_board = (IN_ALPHABET[int(action_b% BOARD_SIZE)],IN_NUMBER[int(action_b // BOARD_SIZE)])

                # 入力手をチェック（基本的に範囲内にある）
                if board.checkIN(action_board):
                    x = IN_ALPHABET.index(action_board[0]) + 1
                    y = IN_NUMBER.index(action_board[1]) + 1

                # 手を打つ（相手が打ってから、経験を保存する。）
                if not board.move(x, y):
                    continue
                else:
                    # 状態 s′ と報酬rの観測
                    state_next_a = trans(board.RawBoard)
                    if Corner_flag_a == 1:
                        reward_a = 0.2
                    elif Bad_X_hand_flag_a == 1:
                        reward_a = -0.1
                    elif Bad_C_hand_flag_a == 1:
                        reward_a = -0.1
                    else:
                        reward_a = 0
                    if board.Turns > 0:# state,actionがないので、random初手は経験の保存無し
                        memory_a.push(state_a,action_a,state_next_a,reward_a)

                    # 有効手が角であった時にflagを立て、相手行動時に+報酬を与える。
                    Corner_flag_b = 0
                    if x == 1 and y == 1:
                        Corner_flag_b = 1
                    if x == 1 and y == BOARD_SIZE:
                        Corner_flag_b = 1
                    if x == BOARD_SIZE and y == 1:
                        Corner_flag_b = 1
                    if x == BOARD_SIZE and y == BOARD_SIZE:
                        Corner_flag_b = 1
                    # 有効手が悪手:Xであった時に、マイナス報酬
                    Bad_X_hand_flag_b = 0
                    if x == 2 and y == 2:
                         Bad_X_hand_flag_b = 1
                    if x == 2 and y == BOARD_SIZE - 1:
                         Bad_X_hand_flag_b = 1
                    if x == BOARD_SIZE - 1 and y == 2:
                         Bad_X_hand_flag_b = 1
                    if x == BOARD_SIZE - 1 and y == BOARD_SIZE - 1:
                         Bad_X_hand_flag_b = 1
                    # 有効手が悪手:Cであった時に、マイナス報酬
                    Bad_C_hand_flag_b = 0
                    if x == 1 and y == 2:
                        Bad_C_hand_flag_b = 1
                    if x == 2 and y == 1:
                        Bad_C_hand_flag_b = 1
                    if x == 1 and y == BOARD_SIZE - 1:
                        Bad_C_hand_flag_b = 1
                    if x == 2 and y == BOARD_SIZE:
                        Bad_C_hand_flag_b = 1
                    if x == BOARD_SIZE - 1 and y == 1:
                        Bad_C_hand_flag_b = 1
                    if x == BOARD_SIZE and y == 2:
                        Bad_C_hand_flag_b = 1
                    if x == BOARD_SIZE and y == BOARD_SIZE - 1:
                        Bad_C_hand_flag_b = 1
                    if x == BOARD_SIZE - 1 and y == BOARD_SIZE:
                        Bad_C_hand_flag_b = 1


            # board.display()

            # 勝敗の判定 
            if board.isGameOver():
                count_black = np.count_nonzero(board.RawBoard[:, :] == BLACK)
                count_white = np.count_nonzero(board.RawBoard[:, :] == WHITE)
                state_next_a = trans(board.RawBoard)
                state_next_b = trans(board.RawBoard)
                dif = count_black - count_white
                if dif > 0:
                    black_win += 1
                    black_win_interval += 1
                    reward_a = 1
                    reward_b = -1
                elif dif < 0:
                    white_win += 1
                    reward_a = -1
                    reward_b = 1
                elif dif == 0:
                    draw += 1
                    reward_a = 0
                    reward_b = 0
                memory_a.push(state_a,action_a,state_next_a,reward_a)
                memory_b.push(state_b,action_b,state_next_b,reward_b)
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
            w2_init_a = copy.deepcopy(w2_a)
            w3_init_a = copy.deepcopy(w3_a)
            w2_init_b = copy.deepcopy(w2_b)
            w3_init_b = copy.deepcopy(w3_b)
            # 経験をランダムサンプリング
            memory_num = episode_interval * 5 # 間隔数だけ経験を抜き取る
            data_a = memory_a.sample(memory_num)
            data_b = memory_b.sample(memory_num)
            # Q Networkの学習実行：dataからミニバッチ法でやりたい
            Epoch_Q = 400 # エポック数
            Bach_Size_Q = int(episode_interval / 2)
            for _ in range(0,Epoch_Q):
                Random_index_a = random.sample(range(memory_num), k=memory_num)
                Random_index_b = random.sample(range(memory_num), k=memory_num)
                Batc_count = 0
                for k in range(0, memory_num // Bach_Size_Q):
                    Batc_count = Bach_Size_Q * k
                    for i in range(Batc_count,Bach_Size_Q + Batc_count):
                        buf_data_a = data_a[Random_index_a[i]]
                        buf_data_b = data_b[Random_index_b[i]]

                        # 遷移状態におけるTarget_network出力のmaxQ値を取得
                        buf_a = Target_network_a.ForwardPropagation(buf_data_a.next_state, w2_init_a, w3_init_a)
                        buf_b = Target_network_b.ForwardPropagation(buf_data_b.next_state, w2_init_b, w3_init_b)

                        
                        max_Q_a = buf_a['z3'][0]
                        max_Q_b = buf_b['z3'][0]
                        Q_nextstae_max_a = copy.deepcopy(buf_a['z3'])
                        Q_nextstae_max_b = copy.deepcopy(buf_b['z3'])
                        for ii in range(0,BOARD_SIZE * BOARD_SIZE):
                            Q_nextstae_max_a[ii] = 0
                            Q_nextstae_max_b[ii] = 0
                            if max_Q_a < buf_a['z3'][ii]:
                                max_Q_a = buf_a['z3'][ii]
                            if max_Q_b< buf_b['z3'][ii]:
                                max_Q_b = buf_b['z3'][ii]
                        Q_nextstae_max_a[buf_data_a.action] = buf_data_a.reward + 0.99 * max_Q_a
                        Q_nextstae_max_b[buf_data_b.action] = buf_data_b.reward + 0.99 * max_Q_b
                        y_a = copy.deepcopy(Q_nextstae_max_a)
                        y_b = copy.deepcopy(Q_nextstae_max_b)

                        Y_train_a = np.array(y_a)
                        Y_train_b = np.array(y_b)

                        # 順伝播計算をして、fに係数を辞書る。
                        f_a = Q_network_a.ForwardPropagation(buf_data_a.state,w2_a,w3_a)
                        f_b = Q_network_b.ForwardPropagation(buf_data_b.state,w2_b,w3_b)
                        
                        # 誤差逆伝播法により、bに勾配を辞書る
                        b_a = Q_network_a.BackPropagation(Y_train_a,w2_a,w3_a,f_a['z1'],f_a['z2'],f_a['z3'],f_a['u2'])
                        b_b = Q_network_b.BackPropagation(Y_train_b,w2_b,w3_b,f_b['z1'],f_b['z2'],f_b['z3'],f_b['u2'])

                        # 勾配に基づいて重みを更新
                        epsilon = 0.1 # 時によっては、ε-greedy行動選択と同じ値を使ってた。バッチサイズを合わせて調整。0.1くらいがいいかも
                        w2_a = copy.deepcopy(w2_a - epsilon*b_a['dw2'] / Bach_Size_Q)
                        w3_a = copy.deepcopy(w3_a - epsilon*b_a['dw3'] / Bach_Size_Q)
                        d3_a = b_a['d3']
                        w2_b = copy.deepcopy(w2_b - epsilon*b_b['dw2'] / Bach_Size_Q)
                        w3_b = copy.deepcopy(w3_b - epsilon*b_b['dw3'] / Bach_Size_Q)
                        d3_b = b_b['d3']

                        error_a.append(max(max(abs(d3_a), key=max)))
                        error_b.append(max(max(abs(d3_b), key=max)))
                        #print(w['w2'])


            print('Interval Learning -- episodes:' + str(episode))
            print('black_win_interval_rate：' + str(black_win_interval / episode_interval))
            print('loss max a：' + str(max(max(abs(d3_a), key=max))))
            print('loss max b：' + str(max(max(abs(d3_b), key=max))))

            # 経験の初期化
            memory_a.__init__(400)
            memory_b.__init__(400)
            black_win_rate.append(black_win_interval / episode_interval)

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
    x = range(len(error_a))
    plt.plot(x,error_a,color = 'blue', marker = 'v')
    plt.plot(x,error_b,color = 'orange', marker = 'o')
    # plt.savefig("black rate.png")
    # 経験を学習する毎における、学習時の誤差の最大値を表示
    plt.show()
    #------------------------結果の可視化------------------------

    # 重みの保存
    np.save("lsi_reverse/weight/w2_test_a",w2_a)
    np.save("lsi_reverse/weight/w3_test_a",w3_a)
    np.save("lsi_reverse/weight/w2_test_b",w2_b)
    np.save("lsi_reverse/weight/w3_test_b",w3_b)

if __name__ == '__main__':

    to_osero()