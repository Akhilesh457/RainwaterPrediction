#cleaning the data
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
'''
#read data 
data = pd.read_csv("weather.csv")

#drop the unnecessary columns in the data
data = data.drop(['Events','Date','SeaLevelPressureHighInches'],axis=1)
data = data.replace('T',0.0)
data = data.replace('-',0.0)
data.to_csv('weather_prediction.csv')
'''
#read the cleaned data
data = pd.read_csv("weather_prediction.csv")

X = data.drop(['PrecipitationSumInches'],axis=1)

Y = data['PrecipitationSumInches']
Y = Y.values.reshape(-1,1)
day_index = 798
days = [i for i in range(Y.size)]

clf = LinearRegression()
clf.fit(X,Y)

inp = np.array([[74],[60],[67],[49],[33],[29.68],[4],[31],[54],[96],[41],[12],[30],[89],[52],[78],[15],[61]])

inp = inp.reshape(1,-1)

print("The precipitation in inches is :" , clf.predict(inp))

print('The precipitation trend again:')
plt.scatter(days,Y,color = 'g')
plt.scatter(days[day_index],Y[day_index],color='r')
plt.title('Precipitation level')
plt.xlabel('Days')
plt.ylabel('Precipitation in inches')
plt.show()
x_f = X.filter(['TempAvgF','DewPointAvgF','HumidityAvgPercent','SeaLevelPressureAvgInches','VisibilityAvgMiles','WindAvgMPH'],axis=1)
print('Precipitation VS selected Attributes Graph:')
for i in range(x_f.columns.size):
    plt.subplot(3,2,i+1)
    plt.scatter(days,x_f[x_f.columns.values[i][:100]],color='g')
    plt.scatter(days[day_index],x_f[x_f.columns.values[i]][day_index],color='r')
    plt.title(x_f.columns.values[i])
    plt.tight_layout()
#plot a graph of precipitation levels VS no of days
plt.show()


