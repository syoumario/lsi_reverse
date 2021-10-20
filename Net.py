#参考資料：http://deepblue-ts.co.jp/deep-learning/3layer_nn_fullscratch_2/

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def del_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 順伝播計算：入力から出力までを計算する。
# 入力x：例[1,2,3]
# 重みw2,w3：例それぞれ[[1,2,3],[4,5,6]] ※重みの列数が出力の数
def ForwardPropagation(x,w2,w3):
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
def BackPropagation(y,w2,w3,z1,z2,z3,u2):
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
def decent(x,y,w2,w3,epsilon):
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
def train(X,Y,n2,epoch,epsilon):
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
def predict(x,w2,w3):
    # 学習済み係数で順伝搬計算
    f = ForwardPropagation(x,w2,w3)

    # 辞書f内の出力z3を返す
    return f['z3']

# 実験
def test():
    def f(x):
        return (x-1)**2
    
    # サンプルの入力データ
    X = np.random.uniform(0,2,20)

    # 標準偏差0.1、平均0の微妙な誤差を与えた教師データ
    Y = f(X) + np.random.normal(0,0.1,20)

    plt.scatter(X,Y)
    plt.show

    # 転置させて、縦で観た方がイメージしやすいと思う
    X_train = np.array([X]).T
    Y_train = np.array([Y]).T

    # 学習の実行
    w = train(X_train,Y_train,5,5000,0.01)

    # 要素数が等差数列となる配列を生成
    X_test = np.linspace(0,2,100)

    # X_testの要素数分を予測
    Y_test = [predict(x,w['w2'],w['w3']) for x in X_test]
    plt.plot(X_test,Y_test)
    plt.scatter(X,Y)
    plt.show

if _name_ == '_main_':
    test()