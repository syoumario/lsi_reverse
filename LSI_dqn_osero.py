#参考資料：http://deepblue-ts.co.jp/deep-learning/3layer_nn_fullscratch_2/

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
BOARD_SIZE = 4

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
    memory = ExperienceMemory(100)

    print(trans(board.RawBoard))

    # ------------重みの初期化------------
    n1 = len(IN) # 入力の要素数
    n3 = len(OUT) # 入力の要素数
    n2 = BOARD_SIZE * BOARD_SIZE # 中間層のユニット数
    w2 = np.random.normal(0,1,(n2,n1))
    w2 = np.insert(w2,0,0,axis=1)
    w3 = np.random.normal(0,1,(n3,n2))
    w3 = np.insert(w3,0,0,axis=1)
    # -----------------------------------

    buf_action = 0
    NB_EPISODE = 2000

    error = []

    black_win = 0
    black_win_ep = []
    white_win = 0
    draw = 0
    
    for episode in range(NB_EPISODE):
        while True:
            # 状態を受け取る（経験保存）タイミングは、AG有効手ではない時と、CPが打ち返した時、AG or CPで勝敗がついた時、
            # パスが生じた時の経験保存がまだ
            if board.Turns % 2 == 0: # コンピュータ　== : 黒（先手）,!= 白（後手）
                IN = (IN_ALPHABET[random.randint(0,7)],IN_NUMBER[random.randint(0,7)])
                # 入力手をチェック
                if board.checkIN(IN):
                    x = IN_ALPHABET.index(IN[0]) + 1
                    y = IN_NUMBER.index(IN[1]) + 1
                else:
                    continue

                # 状態のbuff
                IN = trans(board.RawBoard)
                # 手を打つ
                if not board.move(x, y):
                    continue
                else:
                    # 3.3 相手（ＰＣ）手有効時：以前自手（agent）と報酬（0）の保存
                    IN_next = trans(board.RawBoard)
                    rewad = 0
                    action = buf_action
                    # 4.経験e=⟨s,a,s′,r⟩をExperience Bufferに保存：⟨s,a,s′,r⟩の組み合わせ
                    memory.push(IN,action,IN_next,rewad)
                    #print('経験保存')

                # 盤面の表示
                #board.display()
 
                # 終局判定
                if board.isGameOver():
                    #board.display()
                    ## 各色の数
                    count_black = np.count_nonzero(board.RawBoard[:, :] == BLACK)
                    count_white = np.count_nonzero(board.RawBoard[:, :] == WHITE)
                    ## 勝敗判定
                    # 3.4 PC手勝敗時：相手手で打った状態と勝敗による報酬
                    IN_next = trans(board.RawBoard)
                    dif = count_black - count_white
                    if dif > 0:#先手（黒）が勝つ
                        black_win += 1
                        reward = 1
                    elif dif < 0:#後手（白）が勝つ
                        white_win += 1
                        reward = -1
                    elif dif == 0:#引き分け
                        draw += 1
                        reward = 0
                    # 4.経験e=⟨s,a,s′,r⟩をExperience Bufferに保存：⟨s,a,s′,r⟩の組み合わせ
                    memory.push(IN,action,IN_next,rewad)
                    #print('経験保存')

                    break
 
                # パス
                if not board.MovablePos[:, :].any():
                    board.CurrentColor = - board.CurrentColor
                    board.initMovable()
                    print()
                    continue
            
            else: # agent　== : 白（先手）,!= 黒（後手）
                # 0.現在の状態INを観測する
                IN = trans(board.RawBoard)

                # 1.現在の状態SをTarget NetworkQ(s′,a|θ−)に入力
                buf = Target_network.ForwardPropagation(IN, w2, w3)

                # 2.Target Networkから出力されたQ値を元に,ε-greedy選択法で行動選択
                epsilon = 0.1
                if np.random.uniform() < epsilon:
                    action = np.random.randint(0, len(ACT))
                else:   
                    # np.argmax：配列の最大要素数のインデックスを返す
                    # 順伝搬計算の出力(Q値)が最大となるインデックスから、行動番号を選択
                    action = ACT[np.argmax(buf['z3'])]

                #print('行動:0 ~ ' + str(BOARD_SIZE * BOARD_SIZE) + '内の行動選択肢：' + str(action))
    
                # 3.行動して、行動したことによって変化した状態 s′ と報酬rの観測
                # 行動を入力
                action_board = (IN_ALPHABET[int(action % BOARD_SIZE)],IN_NUMBER[int(action // BOARD_SIZE)])
                #action_board = (IN_ALPHABET[random.randint(0,7)],IN_NUMBER[random.randint(0,7)])

                # 入力手をチェック（基本的に範囲内にある）
                if board.checkIN(action_board):
                    x = IN_ALPHABET.index(action_board[0]) + 1
                    y = IN_NUMBER.index(action_board[1]) + 1

                # 手を打つ
                if not board.move(x, y):
                    # #3.1 非有効手時:状態は変わらず、マイナス報酬を記録
                    # IN_next = trans(board.RawBoard)
                    # reward = 0
                    # # （意味ない）報酬rを報酬値rを-1〜1の範囲にクリップします。単純な方法としては1以上であれば1に。-1以下であれば-1に報酬をクリップします。
                    # if reward > 1:
                    #     reward = 1
                    # elif reward < -1:
                    #     reward = -1
                    # # 4.経験e=⟨s,a,s′,r⟩をExperience Bufferに保存：⟨s,a,s′,r⟩の組み合わせ
                    # memory.push(IN,action,IN_next,reward)

                    continue
                else:
                    buf_action = action # CP側に行動を渡す
                

                # 盤面の表示
                #board.display()

                # 終局判定
                if board.isGameOver():
                    #board.display()
                    ## 各色の数
                    count_black = np.count_nonzero(board.RawBoard[:, :] == BLACK)
                    count_white = np.count_nonzero(board.RawBoard[:, :] == WHITE)
                    ## 勝敗判定
                    # 3.2 自手勝敗時：自手で打った状態と勝敗による報酬
                    IN_next = trans(board.RawBoard)
                    dif = count_black - count_white
                    if dif > 0:#先手（黒）が勝つ
                        black_win += 1
                        reward = 1
                    elif dif < 0:#後手（白）が勝つ
                        white_win += 1
                        reward = -1
                    elif dif == 0:#引き分け
                        draw += 1
                        reward = 0
                    # 4.経験e=⟨s,a,s′,r⟩をExperience Bufferに保存：⟨s,a,s′,r⟩の組み合わせ
                    memory.push(IN,action,IN_next,reward)
                    #print('経験保存')

                    break

                 # パス
                if not board.MovablePos[:, :].any():
                    board.CurrentColor = - board.CurrentColor
                    board.initMovable()
                    #print('パスしました')
                    print()
                    continue

     
                # 5.（定期動作）Experience Bufferから任意の経験を取り出し、Q Networkをミニバッチ学習(Experience Replay)

        #ボードの初期化
        board.__init__(BOARD_SIZE)
        # 5.（定期動作）Experience Bufferから任意の経験を取り出し、Q Networkをミニバッチ学習(Experience Replay)
        w2_init = w2 # Target_network用に重みを固定
        w3_init = w3 # Target_network用に重みを固定
        if episode % 100 == 0 and episode != 0:
            # 経験をランダムサンプリング
            memory_num = 80
            data = memory.sample(memory_num)
            # Q Networkの学習実行：dataからミニバッチ法でやりたい
            Epoch_Q = 200
            Bach_Size_Q = 40
            for _ in range(Epoch_Q):
                Random_index = random.sample(range(memory_num), k=Bach_Size_Q)
                for i in range(0,Bach_Size_Q):
                    buf_data = data[Random_index[i]]
                    # buf = Q_network.ForwardPropagation(buf_data.state, w2, w3)
                    # # x = buf['z3'][buf_data.action]
                    # x = buf['z3']
                    # print(x)
                    buf = Target_network.ForwardPropagation(buf_data.next_state, w2_init, w3_init)
                    # max意外をゼロにする！
                    max_q = buf['z3'][0]
                    max_act_p = 0
                    for ii in range(0,BOARD_SIZE * BOARD_SIZE):
                        if max_q < buf['z3'][ii]:
                            max_q = buf['z3'][ii]
                            max_act_p = ii
                    for ii in range(0,BOARD_SIZE * BOARD_SIZE):
                        if ii != max_act_p:
                            buf['z3'][ii] = 0
                        else:
                            buf['z3'][ii] = buf_data.reward + 0.9 * buf['z3'][ii]
                        Q_nextstae_max = buf['z3']

                    y = Q_nextstae_max
                    #print(np.array([y]).T)

                    X_train = np.array(buf_data.state).T
                    Y_train = np.array(sum(y)).T
                    # print(X_train)
                    # print(Y_train)

                    # Ｑネットワーク：順伝播計算はあっていると思うんで、誤差逆伝播法と確率的勾配降下法のところを新たに作る！
                    # 入力：状態、出力：Ｑ値は同じに、誤差を教師でｙ－ｘが最小になるように重みを１ステップ更新する式
                    w = Q_network.decent(X_train,Y_train,w2,w3,epsilon)
                    w2 = w['w2']
                    w3 = w['w3']
                    d3 = w['d3']
                    error.append(max(d3))
                    
                    # print(w['w2'])
            w2_init = w2
            w3_init = w3
            print('経験より学習しました。今のエピソードは' + str(episode))

            #r = dict(w2=w2,w3=w3)
            # 経験の初期化
            memory.__init__(100)
            black_win_ep.append(black_win)
            
    print('Black WIN：' + str(black_win))
    print('White WIN：' + str(white_win))
    print('draw：' + str(draw))
    print(black_win_ep)
    x = range(len(black_win_ep))
    plt.plot(x,black_win_ep)
    # 傾きが変化していれば、学習していると思う。報酬の与え方で、上か下か
    plt.show()
    x = range(len(error))
    plt.plot(x,error)
    # 経験を学習する毎における、学習時の誤差の最大値を表示
    plt.show()

if __name__ == '__main__':
    #test()

    to_osero()
    
