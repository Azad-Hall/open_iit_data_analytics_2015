import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

out_of_stock = []
def formula(s):
	ans = [0, 0, 0]
	avg=(s[0]+s[1]+s[2])/3.0
	count=0

	for i in range(3,len(s)):
		j=i-3
		avg=avg+(s[i]-s[j])/3.0
		deviation=0.0
		for k in range(-1,2):
			if(i+k<len(s)):
				deviation=deviation+(avg-s[i+k])*(avg-s[i+k])
		deviation=deviation/3.0
		deviation=math.sqrt(deviation)
		#avg_array.append(float(avg))
		#lower_bound_array.append(float(avg-deviation+0.001))
		#upper_bound_array.append(float(avg+deviation))
		if(s[i]<float(avg-deviation)):
			ans.append(1)
		else:
			ans.append(0)
	
	out_of_stock.append(ans)	

df = pd.read_csv('da.csv')
df = df[df.Flag != 'Forecast']
gf = df.groupby([ 'City', 'Department', 'Product'])

grouped = []
names = []
sums = 0

for name, group in gf:
    names.append(name)
    sl = group['Sale']
    sl = [int(x) for x in sl]
    grouped.append(sl)

for i in range(len(grouped)):
	sums += len(grouped[i])
	formula(grouped[i])
sums2 = 0;
for i in range(len(out_of_stock)):
	#print(len(out_of_stock[i]))
	sums2 += len(out_of_stock[i])
print(sums2)
print(sums)	
df = pd.read_csv('da.csv')
df.insert(0, "Out of Stock", 0)
idx = 0
#print(df['Flag'][32])
for i in range(len(out_of_stock)):
	while df['Flag'][idx] =='Forecast':
		df['Out of Stock'][idx] = 'no answer'
		idx += 1
		#print(idx)
	k = 0
	for l in range(len(out_of_stock)):
		if names[l][0] == df['City'][idx] and names[l][1] == df['Department'][idx] and names[l][2] == df['Product'][idx] :
			k = l;
			break;
	for j in range(len(out_of_stock[k])):
		#print(idx)
		df['Out of Stock'][idx] = out_of_stock[k][j]
		idx += 1
df.to_csv("out_of_stock.csv")

'''
lower_bound_array.append(0)
upper_bound_array.append(0)
avg_array.append(0)
lower_bound_array.append(0)
upper_bound_array.append(0)
avg_array.append(0)
plt.plot(range(len(s)),s,'r-',avg_array,'g-',lower_bound_array,'b-',upper_bound_array,'c-')
plt.show()
'''