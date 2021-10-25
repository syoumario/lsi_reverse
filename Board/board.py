import numpy as np
# 環境：self.RawBoard = [[BOARD_SIZE + 2] * (BOARD_SIZE + 2)]
# 行動：x,y = x:1 ~ BOARD_SIZE - 1, y:1 ~ BOARD_SIZE - 1
# 手番数：self.Turns = 0 → 先手黒スタート

# マスの状態
EMPTY = 0 # 空きマス
WHITE = -1 # 白石
BLACK = 1 # 黒石
WALL = 2 # 壁

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
 
class Board:

    def __init__(self,board_size):
        """
        ボードの表現
        """ 
        # ボードのサイズ
        self.BOARD_SIZE = board_size

        # 全マスを空きマスに設定
        self.RawBoard = np.zeros((self.BOARD_SIZE + 2, self.BOARD_SIZE + 2), dtype=int)
 
        # 壁の設定
        self.RawBoard[0, :] = WALL
        self.RawBoard[:, 0] = WALL
        self.RawBoard[self.BOARD_SIZE + 1, :] = WALL
        self.RawBoard[:, self.BOARD_SIZE + 1] = WALL
 
        # 初期配置
        self.RawBoard[self.BOARD_SIZE //2 , self.BOARD_SIZE //2] = WHITE
        self.RawBoard[self.BOARD_SIZE //2 + 1, self.BOARD_SIZE //2 + 1] = WHITE
        self.RawBoard[self.BOARD_SIZE //2, self.BOARD_SIZE //2 + 1] = BLACK
        self.RawBoard[self.BOARD_SIZE //2 + 1, self.BOARD_SIZE //2] = BLACK
 
        # 手番
        self.Turns = 0
 
        # 現在の手番の色
        self.CurrentColor = BLACK
 
        # 置ける場所と石が返る方向
        self.MovablePos = np.zeros((self.BOARD_SIZE + 2, self.BOARD_SIZE + 2), dtype=int)
        self.MovableDir = np.zeros((self.BOARD_SIZE + 2, self.BOARD_SIZE + 2), dtype=int)
 
        # MovablePosとMovableDirを初期化
        self.initMovable()

        # 手数の上限
        self.MAX_TURNS = self.BOARD_SIZE * self.BOARD_SIZE - 4
 
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
        if x < 1 or self.BOARD_SIZE < x:
            return False
        if y < 1 or self.BOARD_SIZE < y:
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
        for x in range(1, self.BOARD_SIZE + 1):
            for y in range(1, self.BOARD_SIZE + 1):
 
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
        if self.Turns >= self.MAX_TURNS:
            return True
 
        # (現在の手番)打てる手がある場合はゲームを終了しない
        if self.MovablePos[:, :].any():
            return False
 
        # (相手の手番)打てる手がある場合はゲームを終了しない
        for x in range(1, self.BOARD_SIZE + 1):
            for y in range(1, self.BOARD_SIZE + 1):
 
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
        for y in range(1, self.BOARD_SIZE + 1):
 
            # 縦軸
            print(y, end="")
            # 横軸方向へのマスのループ
            for x in range(1, self.BOARD_SIZE + 1):
 
                # マスの種類(数値)をgridに代入
                grid = self.RawBoard[x, y]
 
                # マスの種類によって表示を変化
                if grid == EMPTY: # 空きマス
                    print('□', end="")
                elif grid == WHITE: # 白石
                    print('●', end="")
                elif grid == BLACK: # 黒石
                    print('◌', end="")
 
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
