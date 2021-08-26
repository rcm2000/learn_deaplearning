import pandas as pd

df1 = pd.read_csv('titanic/submission.csv')
df1.to_csv('gender_submission.csv',index = False)