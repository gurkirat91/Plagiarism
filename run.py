from main import preprocessing,vectorize,similarity,plag_check
import pandas as pd

user_name=input("enter your name")
user_content=input("enter the content")
content=[]
author=[]



df=pd.read_csv("articles1.csv")
df=df.head(5)

df=df[['author','content']]
new_row=pd.DataFrame({'author':user_name,'content':user_content},index=[0])
df=pd.concat([new_row,df.loc[:]]).reset_index(drop=True)
aut=df['author']
df=df['content']
df=preprocessing(df)


for i in df:
    content.append(i)
for i in aut:
    author.append(i)

vectors=vectorize(content)
s_vector=list(zip(author,vectors))


#check for plag.....
scores,sim_author=plag_check(s_vector)
m=max(scores)
aut=scores.index(m)
print(f"max plag with {sim_author[aut]} = {scores[aut]}")






