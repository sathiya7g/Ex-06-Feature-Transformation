# Ex-06 Feature Transformation

## AIM
To read the given data and perform Feature Transformation process and save the data to a file.

## ALGORITHM
### STEP 1

Read the given Data

### STEP 2

Clean the Data Set using Data Cleaning Process

### STEP 3

Apply Feature Transformation techniques to all the feature of the data set

### STEP 4

Save the data to the file

## CODE

import pandas as pd

df=pd.read_csv('/content/Data_to_Transform.csv')

df.head()

df.isnull().sum()

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')

plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')

plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')

plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')

plt.show()

df['Highly Positive Skew']=np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')

plt.show()

df['Highly Positive Skew']=1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew']=np.sqrt(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')

plt.show()

from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer("yeo-johnson")

df['Moderate Negative Skew']=pd.DataFrame(pt.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')

plt.show()

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df['Moderate Negative Skew']=pd.DataFrame(pt.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')

plt.show()



## OUTPUT

![op1](https://user-images.githubusercontent.com/112301582/233019538-957ad6bb-75f1-4ddc-95e5-809f1c58294e.png)

![op2](https://user-images.githubusercontent.com/112301582/233019568-fe8890db-39a2-4983-b95d-ac59bfd2b882.png)

![op3](https://user-images.githubusercontent.com/112301582/233019599-36324a7e-6f28-4140-8b26-301b8b6c9f39.png)

![op4](https://user-images.githubusercontent.com/112301582/233019631-ad209ce4-981a-4016-b706-ea0e30650a70.png)

![op5](https://user-images.githubusercontent.com/112301582/233019700-ad5ca9a9-5488-4b46-bd2c-da4bde84be67.png)

![op6](https://user-images.githubusercontent.com/112301582/233019801-b1576933-a191-4f2a-ba64-affeffb8f911.png)

![op7](https://user-images.githubusercontent.com/112301582/233019857-8a51d18b-ecb4-4349-9628-459351d5b201.png)

![op8](https://user-images.githubusercontent.com/112301582/233019938-36d96b49-2dc6-4f03-841a-1d9e0aff4347.png)

![op9](https://user-images.githubusercontent.com/112301582/233019988-892a04e7-b7af-47b9-a52b-709e9e491bf6.png)

![op10](https://user-images.githubusercontent.com/112301582/233020043-b6bdbb5d-8a22-4100-b188-af63217dc34d.png)

![op11](https://user-images.githubusercontent.com/112301582/233020114-a61ca978-cae8-46a3-a229-1fa0951a1353.png)

## RESULT
Thus feature transformation on the given dataset was performed and results were stored to dataset.
