from tqdm import tqdm
import time

# 创建一个包含10个元素的列表
my_list = list(range(10))
# 使用tqdm迭代列表并显示进度条
for item in tqdm(my_list, mininterval=0):
    # 模拟一些处理时间
    time.sleep(0.1)