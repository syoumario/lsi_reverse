#参考資料：http://deepblue-ts.co.jp/deep-learning/3layer_nn_fullscratch_2/

import numpy as np
import matplotlib.pyplot as plt
import random
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


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def del_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

class net():

    # 順伝播計算：入力から出力までを計算する。
    # 入力x：例[1,2,3]
    # 重みw2,w3：例それぞれ[[1,2,3],[4,5,6]] ※重みの列数が出力の数
    def ForwardPropagation(self,x,w2,w3):
        # 重みと内積するために、転置させて一列にする
        # 分からんけど、最初の行に新たに行[1,1,..]を挿入する。-----なぜかを聞く！
        z1 = np.insert(np.array([x]).T,0,1,axis=0)

        #　重みと入力の内積をとる。出力例：[?,?]の１行
        u2 = np.dot(w2,z1)

        # シグモイド関数でu2:0~1に変換して、転置させて１列にする
        z2 = np.insert(sigmoid(u2),0,1,axis=0)

        # 重みとの内積をとる。出力例：[?,?]の１行
        u3 = np.dot(w3,z2)

        z3 = u3

        # 辞書生成：{'z1':z1,'z2':z2,'z3':z3,'u2':u2}
        return dict(z1=z1,z2=z2,z3=z3,u2=u2)

    # 誤差逆伝播法：順伝搬計算の返値から、誤差関数の微分を返す
    # 教師ｙ：出力と同様な形のリストor配列：例：[?,?]
    # 他：各係数行列
    def BackPropagation(self,y,w2,w3,z1,z2,z3,u2):
        # 出力と教師の誤差を計算
        d3 = (z3 - np.array([y]).T).T

        #　シグモイドの場合の誤差逆伝播法における勾配式
        d2 = np.dot(d3,w3)[:,1:]*del_sigmoid(u2).T

        #
        dw3 = d3.T*z2.T

        #
        dw2 = d2.T*z1.T

        # 勾配を辞書で返す
        return dict(dw2=dw2,dw3=dw3)

    # 確率的勾配降下法：誤差逆伝播法で計算された計数行列の勾配を利用して計数の更新を行う
    def decent(self,x,y,w2,w3,epsilon):
        # 順伝播計算をして、fに係数を辞書る。
        f = ForwardPropagation(x,w2,w3)

        # 誤差逆伝播法により、bに勾配を辞書る
        b = BackPropagation(y,w2,w3,f['z1'],f['z2'],f['z3'],f['u2'])

        # 勾配に基づいて重みを更新
        # 学習率epsilon：係数をどの程度更新するか
        w2 = w2 - epsilon*b['dw2']
        w3 = w3 - epsilon*b['dw3']

        # １順で更新された重みを辞書で返す
        return dict(w2=w2,w3=w3)

    # 係数の初期化と学習の実行
    # 入力X,出力Y：リストor配列：例[[1,2,3],[2,3,4],...]
    # n2：中間層のユニット数：ニューロン数：
    def train(self,X,Y,n2,epoch,epsilon):
        # 教師の１行の列数分の長さ：例では3
        n1 = len(X[0])
        n3 = len(Y[0])

        # 教師の列数、ユニット数に合わせた次元の重みを初期値（平均０、標準偏差１の正規分布）で作る
        w2 = np.random.normal(0,1,(n2,n1))

        # 各行の列の最初に、列を引き延ばして０を入れる。-----上と合わせてなぜかを聞く！
        w2 = np.insert(w2,0,0,axis=1)

        w3 = np.random.normal(0,1,(n3,n2))
        w3 = np.insert(w3,0,0,axis=1)

        # 学習実行
        for _ in range(epoch):
            for x,y in zip(X,Y):
                w = decent(x,y,w2,w3,epsilon)
                w2 = w['w2']
                w3 = w['w3']

        # 学習結果を辞書で返す
        return dict(w2=w2,w3=w3)

    # 学習済み係数を用いて予測
    def predict(self,x,w2,w3):
        # 学習済み係数で順伝搬計算
        f = ForwardPropagation(x,w2,w3)

        # 辞書f内の出力z3を返す
        return f['z3']

# 実験
    # def test():
    #     def f(x):
    #         return (x-1)**2
        
    #     # サンプルの入力データ
    #     X = np.random.uniform(0,2,50)

    #     # 標準偏差0.1、平均0の微妙な誤差を与えた教師データ
    #     Y = f(X) + np.random.normal(0,0.1,50)
        
    #     # 転置させて、縦で観た方がイメージしやすいと思う
    #     X_train = np.array([X]).T
    #     Y_train = np.array([Y]).T

    #     # 学習の実行
    #     w = train(X_train,Y_train,5,5000,0.01)
    #     print('学習終了')

    #     # 要素数が等差数列となる配列を生成
    #     X_test = np.linspace(0,2,100)

    #     # X_testの要素数分を予測
    #     Y_test = [predict(x,w['w2'],w['w3']) for x in X_test]
    #     plt.plot(X_test,Y_test)
    #     plt.show()

# class Board() のRawBoardをIN = [0] * (BOARD_SIZE * BOARD_SIZE)の状態に変換
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

    Target_network = net()
    Q_network = net()
    board = Board() 

    print(trans(board.RawBoard))

    # ------------重みの初期化------------
    n1 = len(IN) # 入力の要素数
    n3 = len(OUT) # 入力の要素数
    n2 = 5 # 中間層のユニット数
    w2 = np.random.normal(0,1,(n2,n1))
    w2 = np.insert(w2,0,0,axis=1)
    w3 = np.random.normal(0,1,(n3,n2))
    w3 = np.insert(w3,0,0,axis=1)
    # -----------------------------------

    NB_EPISODE = 3000
    for episode in range(NB_EPISODE):
        while True:
            
            if board.Turns % 2 != 0: # コンピュータ　== : 黒（先手）,!= 白（後手）
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

                print('行動:0 ~ ' + str(BOARD_SIZE * BOARD_SIZE) + '内の行動選択肢：' + str(action()))
    
                # 3.行動して、行動したことによって変化した状態 s′ と報酬rの観測
                # 行動を入力
                action_board = (IN_ALPHABET[int(action % BOARD_SIZE)],IN_NUMBER[int(action // BOARD_SIZE)])

                # 入力手をチェック（基本的に範囲内にある）
                if board.checkIN(action_board):
                    x = IN_ALPHABET.index(action_board[0]) + 1
                    y = IN_NUMBER.index(action_board[1]) + 1

                # 手を打つ
                if not board.move(x, y):
                    continue
                

                # 盤面の表示
                #board.display()

                # 終局判定
                if board.isGameOver():
                    #board.display()
                    ## 各色の数
                    count_black = np.count_nonzero(board.RawBoard[:, :] == BLACK)
                    count_white = np.count_nonzero(board.RawBoard[:, :] == WHITE)
                    ## 勝敗判定
                    dif = count_black - count_white
                    if dif > 0:#先手（黒）が勝つ
                        black_win += 1
                    elif dif < 0:#後手（白）が勝つ
                        white_win += 1
                    elif dif == 0:#引き分け
                        draw += 1

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

                # INが更新される
                # 報酬：rewardを得る

                IN_next = [0] * (BOARD_SIZE * BOARD_SIZE) # 本当は変化した状態 s′を入れたい
                reward = 10 # 本当は得られた報酬rを入れたい

                # 4.経験e=⟨s,a,s′,r⟩をExperience Bufferに保存
                # 報酬rを報酬値rを-1〜1の範囲にクリップします。単純な方法としては1以上であれば1に。-1以下であれば-1に報酬をクリップします。
                if reward > 1:
                    reward = 1
                elif reward < -1:
                    reward = -1
                
                # Experience Bufferに保存：⟨s,a,s′,r⟩の組み合わせだが、どうやって？


                # 5.（定期動作）Experience Bufferから任意の経験を取り出し、Q Networkをミニバッチ学習(Experience Replay)



    #ボードの初期化
    board.__init__()
    



    def f(x):
        return (x-1)**2
    
    # サンプルの入力データ
    X = np.random.uniform(0,2,50)

    # 標準偏差0.1、平均0の微妙な誤差を与えた教師データ
    Y = f(X) + np.random.normal(0,0.1,50)
    
    # 転置させて、縦で観た方がイメージしやすいと思う
    X_train = np.array([X]).T
    Y_train = np.array([Y]).T

    
    # --------------学習の実行--------------
    epoch = 200
    epsilon = 0.01
    n1 = len(X_train[0])
    n3 = len(Y_train[0])
    w2 = np.random.normal(0,1,(n2,n1))
    w2 = np.insert(w2,0,0,axis=1)
    w3 = np.random.normal(0,1,(n3,n2))
    w3 = np.insert(w3,0,0,axis=1)
    for _ in range(epoch):
        for x,y in zip(X_train,Y_train):
            w = decent(x,y,w2,w3,epsilon)
            w2 = w['w2']
            w3 = w['w3']
    r = dict(w2=w2,w3=w3)
    # --------------学習の実行--------------

    # 要素数が等差数列となる配列を生成
    X_test = np.linspace(0,2,100)

    # X_testの要素数分を予測
    Y_test = [predict(x,r['w2'],w['w3']) for x in X_test]
    plt.plot(X_test,Y_test)
    plt.show()

if __name__ == '__main__':
    #test()

    to_osero()
    
