import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
sales=pd.read_csv("nis_test.csv")
s=sales.values
avg=(s[0]+s[1]+s[2])/3.0
dynamic_window=0
flag_in1=0
flag_in2=1
flag_in3=2
dynamic_window_array=list()
avg_array=list()
dynamic_window_array.append(0)
avg_array.append(0)
dynamic_window_array.append(0)
avg_array.append(0)
dynamic_window_array.append(0)
avg_array.append(0)
dynamic_window_array.append(dynamic_window)
is_out_of_stock=list()
is_out_of_stock.append(0)
is_out_of_stock.append(0)
is_out_of_stock.append(0)
avg_array.append(avg)
consec_out=0
for i in range(3,s.size-1):
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
		if(poisson.pmf(s[i],dynamic_window)<0.018):
			is_out_of_stock.append(5)
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
	dynamic_window_array.append(float(dynamic_window))
	avg_array.append(float(avg))
	if(is_out_of_stock[i]==5):
		print 1
	else:
		print 0

plt.plot(range(len(s)),s,'r-',avg_array,'g-',dynamic_window_array,'b-', is_out_of_stock,'b*')
plt.show()
