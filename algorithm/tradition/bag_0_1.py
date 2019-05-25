"""
0 - 1 背包问题
解决办法：声明一个 大小为  m[n][c] 的二维数组，m[ i ][ j ] 表示 在面对前i件物品，且背包容量为j 时所能获得的最大价值 ，那么我们可以很容易分析得出 m[i][j] 的计算方法，
（1）. j < w[i] 的情况，这时候背包容量不足以放下第 i 件物品，只能选择不拿
m[ i ][ j ] = m[ i-1 ][ j ]
（2）. j>=w[i] 的情况，这时背包容量可以放下第 i 件物品，我们就要考虑拿这件物品是否能获取更大的价值。
如果拿取，m[ i ][ j ]=m[ i-1 ][ j-w[ i ] ] + v[ i ]。 这里的m[ i-1 ][ j-w[ i ] ]指的就是考虑了i-1件物品，背包容量为j-w[i]时的最大价值，也是相当于为第i件物品腾出了w[i]的空间。
如果不拿，m[ i ][ j ] = m[ i-1 ][ j ] , 同（1）
究竟是拿还是不拿，自然是比较这两种情况那种价值最大。
"""

import sys
import numpy as np

value_arr = [0,8,10,6,3,7,2]
weight_arr = [0,4,6,2,2,5,1]
bag_capacity = 12

prog = np.zeros((7,bag_capacity + 1))



for i in range(1,len(value_arr)):
    for j in range(1,bag_capacity + 1):
        if j < weight_arr[i]:
            prog[i][j] = prog[i-1][j]
        else:
            prog[i][j] = max(prog[i-1][j],prog[i-1][j-weight_arr[i]] + value_arr[i])

print(prog)