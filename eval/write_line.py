import matplotlib.pyplot as plt
names = ['One-time-attention', '1', '2', '3', '4']
x = [2,3,4,5,6]
y1 = [0.7733, 0.7597, 0.7597, 0.7613, 0.7615]
y2 = [0.5487, 0.5630, 0.5673, 0.5666, 0.5540]
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11)  # 限定横轴的范围
#pl.ylim(-1, 110)  # 限定纵轴的范围
plt.subplot(211)
plt.xticks(x, names)
plt.plot(x, y1, marker='o', color='red', label=u'Statue AUC')
plt.legend()
plt.subplot(212)
plt.xticks(x, names)
plt.plot(x, y2, marker='*', color='blue', label=u'AUC Micro')
plt.legend()  # 让图例生效
plt.xlabel("Depth") #X轴标签

plt.show()