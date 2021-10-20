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
    #　重みと内積するために、転置させて一列にする
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

    # 
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

    # 重みを辞書で返す
    return dict(w2=w2,w3=w3)