

import pandas as pd
import requests
import re
from PyPDF2 import PdfFileReader
import glob
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train1 = pd.read_csv (r'train1.csv')

train2 = pd.read_csv (r'train2.csv',delimiter=";")

train = pd.concat([train1,train2])

response1 = requests.get("http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/first")

train3 = response1.json()

response2=requests.get('http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/second')

train4 = response2.json()

response3=requests.get('http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/third')

train5 = response3.json()

d1 = pd.DataFrame.from_dict(train3, orient='columns')

train=pd.concat([train,d1])

d2 = pd.DataFrame.from_dict(train4, orient='columns')

train=pd.concat([train,d2])

d3 = pd.DataFrame.from_dict(train5, orient='columns')

train=pd.concat([train,d3])

train.drop('', inplace=True, axis=1)

for file in list(glob.glob('train6/*.pdf')):
  reader = PdfFileReader(file)
  page = reader.pages[0]
  text = page.extractText()
  text = re.sub(" ","",text)
  text= re.sub(",",".",text)
  palabras = re.split("\n",text)
  lista = []
  for i in range(len(palabras)):
    if re.search("COUNTRY",palabras[i]) != None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1): 
      lista.append(NaN)
  for i in range(len(palabras)):
    if re.search("eprtrSectorName",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1):
      lista.append(NaN)
  lista.append(np.nan)
  for i in range(len(palabras)):
    if re.search("FacilityInspireID",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1):
      lista.append(NaN)
  for i in range(len(palabras)):
    if re.search("FACILITYNAME",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1):
      lista.append(NaN)
  for i in range(len(palabras)):
    if re.search("CITY:",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1):
      lista.append(NaN)
  for i in range(len(palabras)):
    if re.search("targetRealase",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1):
      lista.append(NaN)
  for i in range(len(palabras)):
    if re.search("pollutant",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1): 
      lista.append(NaN)
  for i in range(len(palabras)):
    if re.search("YEAR",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1): 
      lista.append(NaN)
  for i in range(len(palabras)):
    if re.search("MONTH",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1): 
      lista.append(NaN)
  for i in range(len(palabras)):
    if re.search("DAY",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1): 
      lista.append(NaN)
  for i in range(len(palabras)):
    if re.search("CONTINENT",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1): 
      lista.append(NaN)
  for i in range(len(palabras)):
    if re.search("max_wind_speed",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1): 
      lista.append(NaN)
  for i in range(len(palabras)):
    if re.search("avg_wind_speed",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1): 
      lista.append(NaN)
  for i in range(len(palabras)):
    if re.search("min_wind_speed",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1): 
      lista.append(NaN)
  for i in range(len(palabras)):
    if re.search("max_temp",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1): 
      lista.append(NaN)
  for i in range(len(palabras)):
    if re.search("avg_temp",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1): 
      lista.append(NaN)
  for i in range(len(palabras)):
    if re.search("min_temp",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1): 
      lista.append(NaN)
  for i in range(len(palabras)):
    if re.search("DAYSFOG",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1): 
      lista.append(NaN)
  for i in range(len(palabras)):
    if re.search("REPORTERNAME",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1): 
      lista.append(NaN)
  for i in range(len(palabras)):
    if re.search("CITY_ID",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1): 
      lista.append(NaN)
  lista.append(np.nan)
  for i in range(len(palabras)):
    if re.search("EPRTRSectorCode",palabras[i])!= None:
      lista.append(palabras[i+1])
      break
    elif i == (len(palabras)-1): 
      lista.append(NaN)
  train.loc[len(train)] = lista

train_pred= train['pollutant']

train_pred=list(map(lambda x: x.replace('Nitrogen oxides (NOX)','0'),train_pred))
train_pred=list(map(lambda x: x.replace('Carbon dioxide (CO2)','1'),train_pred))
train_pred=list(map(lambda x: x.replace('Methane (CH4)','2'),train_pred))

train.drop('pollutant', inplace=True, axis=1)

train=train.applymap(lambda x: x.replace(" ","") if isinstance(x, str) else x)

train.drop('targetRelease',inplace=True, axis = 1) #the same always

train.drop('CONTINENT',inplace=True, axis = 1) #the same always

train.drop('FacilityInspireID',inplace=True, axis = 1)

train.drop('REPORTER NAME',inplace=True, axis=1)

train.drop('City',inplace=True, axis=1)

train.drop('EPRTRAnnexIMainActivityCode',inplace=True, axis=1)

train.drop('EPRTRSectorCode',inplace=True, axis=1) # missing observations

train.drop('DAY',inplace=True, axis=1)

train.drop('MONTH',inplace=True, axis=1)

train.drop('countryName',inplace=True,axis=1)

rf = RandomForestClassifier(n_estimators=50, oob_score=True, random_state=123)
train = pd.get_dummies(train, prefix=[ 'eprtrSectorName','EPRTRAnnexIMainActivityLabel','facilityName','CITY ID'], columns=['eprtrSectorName','EPRTRAnnexIMainActivityLabel','facilityName','CITY ID'])

X_train, X_test, y_train, y_test = train_test_split(train, train_pred, test_size=0.1, random_state=123)


rf.fit(X_train, y_train)
predicted = rf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print(accuracy)

test = pd.read_csv (r'test_x.csv')
test_index=test['test_index']
test.drop('test_index',inplace=True, axis = 1)

test=test.applymap(lambda x: x.replace(" ","") if isinstance(x, str) else x)

test.drop('targetRelease',inplace=True, axis = 1) #the same always

test.drop('CONTINENT',inplace=True, axis = 1) #the same always

test.drop('FacilityInspireID',inplace=True, axis = 1)

test.drop('REPORTER NAME',inplace=True, axis=1)

test.drop('EPRTRAnnexIMainActivityCode',inplace=True, axis=1)

test.drop('EPRTRSectorCode',inplace=True, axis=1) # missing observations

test.drop('DAY',inplace=True, axis=1)

test.drop('MONTH',inplace=True, axis=1)

test.drop('City',inplace=True, axis=1)

test.drop('countryName',inplace=True,axis=1)

test = pd.get_dummies(test, prefix=['eprtrSectorName','EPRTRAnnexIMainActivityLabel','facilityName','CITY ID'], columns=['eprtrSectorName','EPRTRAnnexIMainActivityLabel','facilityName','CITY ID'])
for i in test.columns:
  if i not in train.columns:
    test.drop(i,inplace=True,axis=1)
for i in train.columns:
  if i not in test.columns:
    test[i]=0

predictions = rf.predict(test)

df = pd.DataFrame({'test_index': test_index, 'pollutant': predictions}, columns=['test_index', 'pollutant'])
df.to_csv("predictions.csv", encoding='utf-8', index=False)
df.to_json('predictions.json')

