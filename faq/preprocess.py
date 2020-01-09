import pandas as pd

df = pd.read_csv('./data/baoxianzhidao_filter.csv')
df = df[df['is_best']==1]
best_title = df.apply(
    lambda row:row['question'] if row['question'] is not None and
    len(str(row['question'])) > len(str(row['title'])) else row['title'],axis=1
)
df['best_title'] = best_title

translated = pd.read_csv('./data/synonymous_title.tsv',sep='\t',header=0)

merged = df.merge(translated,left_on='best_title',right_on='best_title')
merged[['best_title','reply','translated']].to_csv('./data/preprocessed_synonymous.csv',index=False,sep='\t')