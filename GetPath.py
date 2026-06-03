import os
# 文件目录（使用脚本所在目录）
rootList = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
# 文件总数
number = int(12)
# 循环新建文件夹
for i in range(1, number + 1):
    realPath = os.path.join(rootList, str(i))
    os.makedirs(realPath)
    print(str(i) + " is ok")
