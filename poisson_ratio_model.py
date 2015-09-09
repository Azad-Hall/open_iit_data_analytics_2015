import numpy as np
import pandas as pd
import math

from scipy.stats import poisson
import matplotlib.pyplot as plt
ones =0
zeroes =0 
out_of_stock = []
def formula(s):
	avg=(s[0]+s[1]+s[2])/3.0
	dynamic_window=0
	flag_in1=0
	flag_in2=1
	flag_in3=2
	is_out_of_stock = [0, 0, 0]
	consec_out=0
	for i in range(3,len(s)):
		pre_avg=avg
		if(s[i]>0):
			flag_in1=flag_in2
			flag_in2=flag_in3
			flag_in3=i
			avg=(s[flag_in1]+s[flag_in2]+s[flag_in3])/3.0
		if(s[i]>pre_avg):
			dynamic_window=avg
		if(s[i]<pre_avg):
			if(is_out_of_stock[i-1]==0):
				dynamic_window=dynamic_window+float(avg)
			if(poisson.pmf(s[i],dynamic_window)<0.01):
				is_out_of_stock.append(1)
				ones+=1
				consec_out=consec_out+1
			else:
				dynamic_window=(dynamic_window-s[i])/(1.5+consec_out)
				is_out_of_stock.append(0)
				consec_out=0
		else:
			is_out_of_stock.append(0)
			consec_out=0
		if(s[i-1]==0 and s[i]>0):
			is_out_of_stock[i]=0
		

	out_of_stock.append(is_out_of_stock)	

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

print ones
print zeroes

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