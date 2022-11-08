import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#0에서 20사이의 데이터를 5X3 사이즈로 랜덤으로 생성
df = pd.DataFrame(np.random.randint(0,20,size=(5, 3)), columns = list('ABC'))
print(df,"\n")

#MinMaxScaler객체 생성
scaler = MinMaxScaler()

#MinMaxScaler로 데이터 변환
scaler.fit(df)
df_scaled = scaler.transform(df)

#transform()을 하면 데이터가 numpy array로 변환이 되어 dataframe으로 변환
df_df_scaled = pd.DataFrame(data=df_scaled, columns = list('ABC'))
print(df_df_scaled)