import pandas as pd
#import pandas_profiling as pp
df=pd.read_csv("houseprices_dataset.csv")
#profile=pp.ProfileReport(df)
#profile.to_file("EDA_houseprices.html")
print(df.dtypes)
print(df.shape)
from sklearn.model_selection import train_test_split
y=df['House_price_inlakhs']
X=df.drop('House_price_inlakhs',axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
from sklearn.metrics import mean_squared_error,mean_absolute_error
mse=mean_squared_error(y_pred,y_test)
mae=mean_absolute_error(y_pred,y_test)
print(mse,mae)
import math
rmse=math.sqrt(mse)
print(rmse)
import pickle
pickle.dump(model,open('model.pkl','wb'))
regression=pickle.load(open('model.pkl','rb'))
print(regression.predict([[10,666.74,6]]))
