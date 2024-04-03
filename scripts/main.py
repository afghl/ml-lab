# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    np.random.seed(0)
    A = np.array([
        [2, 0, 1],
        [0, 0, 0],
        [1, 0, 2]])
    P = np.array([
        [1, 1, 0],
        [0, 0, 1],
        [1, -1, 0]])
    print(np.linalg.eig(A)[0])
    print(np.linalg.eig(A)[1])
    # print(P.T @ P)
#     打印P的逆
#     print(np.linalg.inv(P))
#

#     #  打印B的秩
#     print(np.linalg.matrix_rank(B))
#     # x为列向量
#     x = np.array([2, 1, -1]).reshape(-1, 1)
#     # 求Bx
#     # print(B @ x)
#     A = np.array([ # 三阶不满秩的矩阵
#         [1, 1, 0],
#         [1, 1, 0],
#         [0, 0, 4]])
#     print(np.linalg.matrix_rank(A))
#     x2 = np.array([-2, -3, 2]).reshape(-1, 1)
# #     print(A @ x2)
# #
# # #     转置A和B
# #     print("A的转置")
# #     print(A.T)
# #     # print(B.T)
# #     print("求得B的转置的x")
# #     print(B.T @ np.array([2, -7, 3]).reshape(-1, 1))
# #     print(A.T @ np.array([2, -7, 3]).reshape(-1, 1))
#     # 打印 A * B的第一行
#     print(B[0])
#     print(A @ B)
