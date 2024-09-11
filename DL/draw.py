import matplotlib.pyplot as plt  
import numpy as np  
import matplotlib.pyplot as plt  
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定一个支持中文的字体，例如SimHei  
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
  
# 定义词和对应的值（重要性）  
words = ['身亡', '紧急通知', '隔夜', '吃', '大量', '不能', '正', '千万', '上市', '双双']  
values = [  
    8.332571965752322e-07,  
    -7.445433158892456e-10,  
    1.8512760107114445e-06,  
    3.451461907388875e-07,  
    7.025050052789084e-08,  
    -5.796539994662453e-06,  
    -4.613277226894752e-08,  
    3.1757966212353494e-07,  
    -1.4693672767407406e-07,  
    -7.450996264424248e-09  
] 
print("新闻文本: 紧急通知：夫妇双双身亡！正大量上市，千万不能隔夜吃！") 
for i in range(len(words)):
    print(words[i]+":"+str(values[i]))

# 将正值和负值分开，因为柱状图通常不用于表示负值  
positive_values = [v for v in values if v >= 0]  
negative_values = [v for v in values if v < 0]  
  
# 对应的词也分开  
positive_words = [w for w, v in zip(words, values) if v >= 0]  
negative_words = [w for w, v in zip(words, values) if v < 0]  
  
# 由于值很小，我们可以考虑使用对数刻度，但这里直接展示  
# 绘制正值柱状图  
plt.bar(positive_words, positive_values, color='green')  
  
# 绘制负值柱状图（如果希望表示的话，可以使用不同的颜色或位置）  
plt.bar(negative_words, [-v for v in negative_values], color='red', bottom=np.max(positive_values))  
  
# 设置标题和坐标轴标签  
plt.title('Word Importance')  
plt.xlabel('Words')  
plt.ylabel('Importance (Scaled Value)')  
  
# 显示图表  
plt.tight_layout()  # 调整布局以避免重叠  
plt.show()