from sklearn import preprocessing,tree
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import os

def gbdt_to_rule(file_name,init_score,var,lr,gbdt,lossfunction='friedman_mse'): 
  sas=[]
  sas.append('tnscore=0.0;')

  for now,n_tree in enumerate(gbdt.estimators_):
    with open("{}.txt".format(file_name),"w") as f:
      tree.export_graphviz(n_tree[0],feature_names=var,precision=8,out_file=f)
    a=pd.read_table('{}.txt'.format(file_name))
    a.rename(columns={"digraph Tree {":"tree"},inplace=True)
    a1=a[a.tree.str.contains('->')]
    a3=np.array(a1.values)
    a3=np.insert(a3,0,'0 -> 0',axis=0)
    a2=a[a.tree.str.contains('{}'.format(lossfunction))]
    a2['route']=a3

    a2['from_node']=a2.route.str.split(' ',expand=True)[0]
    a2['to_node']=a2.route.str.split(' ',expand=True)[2]
    a2['node']=a2.tree.str.split(' ',0,expand=True)[0]
    a2['from_node']=a2['from_node'].astype(np.int64,inplace=True)
    a2['to_node']=a2['to_node'].astype(np.int64,inplace=True)
    a2['node']=a2['node'].astype(np.int64,inplace=True)

    a2['tree']=a2.tree.str.split('"',0,expand=True)[1]  
    a4=a2.tree.str.split('value',expand=True)
    a2['value']='response'+a4[1]
    a2['tree']=a2.tree.str.split('\\',expand=True)[0]
    a2['leaf']=a2['node']
    a2['leaf']=0
    a2['leaf'][a2.tree.str.contains('{}'.format(lossfunction))]=1
    a2.drop('route',axis=1,inplace=True)
    leaf=a2[a2.leaf==1]
    for i in range(len(leaf)):
      i=leaf.index[i]
      b=leaf.node[i]
      leaf.tree[i]='if'
      while b>0:
        fore_node=a2.from_node[a2.to_node==b].values[0]
        fore_node=a2[a2.to_node==fore_node]
        if b==leaf.node[i]:
          str=' '
        else:
          str=' and '
        if b-fore_node.node.values[0]>1:
          if fore_node.tree.str.contains('>').values[0]:
            sub_rule=fore_node.tree.str.replace('>','<').values[0]
            leaf.tree[i]=leaf.tree[i]+str+sub_rule
          else:
            sub_rule=fore_node.tree.str.replace('<','>').values[0]
            leaf.tree[i]=leaf.tree[i]+str+sub_rule
        else:
          sub_rule=fore_node.tree.values[0]
          leaf.tree[i]+=str+sub_rule
        b=fore_node.node.values[0]
    leaf.tree=leaf.tree+" then "+leaf.value+";"
    s="* tree {} ; \n".format(now)+"\n".join(leaf.tree) +"tnscore=tnscore+response ;\n"
    sas.append(s)
  sas.append('proba=1- 1/(1+exp({}+tnscore*{})) ;'.format(init_score,lr))
  with open("{}.txt".format("gbdt_rule"),"w") as f:
    f.write("\n ".join(sas))

def read_node(file_name,lossfunction='gini'):
  a=pd.read_table('{}.txt'.format(file_name))

  a.rename(index=str,columns={"digraph Tree {":"tree"},inplace=True)
  a.drop(['0','{}'.format(len(a)-1)],axis=0)
  a1=a[a.tree.str.contains('->')]
  a3=np.array(a1.values)
  a3=np.insert(a3,0,'0 -> 0',axis=0)
  a2=a[a.tree.str.contains('{}'.format(lossfunction))]
  a2['route']=a3

  a2['from_node']=a2.route.str.split(' ',expand=True)[0]
  a2['to_node']=a2.route.str.split(' ',expand=True)[2]
  a2['node']=a2.tree.str.split(' ',0,expand=True)[0]
  a2['from_node']=a2['from_node'].astype(np.int64,inplace=True)
  a2['to_node']=a2['to_node'].astype(np.int64,inplace=True)
  a2['node']=a2['node'].astype(np.int64,inplace=True)

  a2['tree']=a2.tree.str.split('"',0,expand=True)[1]



  a4=a2.tree.str.split('[',expand=True)[1]
  a4=a4.str.split(']',expand=True)[0]
  a2['0_num']=a4.str.split(',',expand=True)[0]
  a2['1_num']=a4.str.split(',',expand=True)[1]
  a2['tree']=a2.tree.str.split('\\',expand=True)[0]
  a2['0_num']=a2['0_num'].astype(np.float64,inplace=True)
  a2['1_num']=a2['1_num'].astype(np.float64,inplace=True)
  a2['density']=a2['1_num']/(a2['1_num']+a2['0_num'])
  a2['lift']=a2['density']/a2['density'][0]
  a2['leaf']=a2['node']
  a2['leaf']=0
  a2['leaf'][a2.tree.str.contains('{}'.format(lossfunction))]=1

  a2.drop('route',axis=1,inplace=True)
  a2.to_csv('{}_transform.csv'.format(file_name),index=False)
  return a2


def find_rule(good_node):
  for i in range(len(good_node)):
    tmp=pd.read_csv('{}_transform.csv'.format(good_node.tree_num[i]))
    a=good_node.node[i]
    good_node.tree[i]='if'
    while a>0:
      fore_node=tmp.from_node[tmp.to_node==a].values[0]
      fore_node=tmp[tmp.to_node==fore_node]
      if a==good_node.node[i]:
        str=' '
      else:
        str=' and '
      if a-fore_node.node.values[0]>1:
        if fore_node.tree.str.contains('>').values[0]:
          sub_rule=fore_node.tree.str.replace('>','<').values[0]
          good_node.tree[i]=good_node.tree[i]+str+sub_rule
        else:
          sub_rule=fore_node.tree.str.replace('<','>').values[0]
          good_node.tree[i]=good_node.tree[i]+str+sub_rule
      else:
        sub_rule=fore_node.tree.values[0]
        good_node.tree[i]+=str+sub_rule
      a=fore_node.node.values[0]
      



def plot_confusion_matrix(cm,names,title='Confusion matrix',cmap=plt.cm.Blues):
  plt.imshow(cm,interpolation='nearest',cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks=np.arange(len(names))
  plt.xticks(tick_marks,names,rotation=45)
  plt.yticks(tick_marks,names)
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Prediction label')

  plt.show()

  return plt
  
  
  
  
def encode_numeric_zscore(df,name,mean=None,sd=None):
  if mean==None:
       mean = df[name].mean()
  if sd==None:
    sd = df[name].std()
  df[name]=(df[name]-mean)/sd
  
  
def encode_numeric_range(df,name,normalized_low=-1,normalized_high=1,
                        data_low=None,data_high=None):
  if data_low == None:
    data_low = min(df[name])
  if data_high == None:
    data_high = max(df[name])
  df[name] = ((df[name]-data_low)/(data_high-df[name]))\
            *(normalized_high-normalized_low)+normalized_low
    
    
def PlotKS(pred,label,n,asc):
  #pred is score:asc=1
  #pred is prob :asc=0
  
  bad = label
  ksds = pd.DataFrame({'bad':bad,'pred':pred})
  ksds['good']=1-ksds.bad
  
  
  if asc==1:
    ksds1 = ksds.sort_values(by=['pred','bad'],ascending=[True,True])
  elif asc==0:
    ksds1 = ksds.sort_values(by=['pred','bad'],ascending=[False,True])
  
  ksds1.index=range(len(ksds1.pred)) # 重置索引为从0开始
  ksds1['cumsum_good1'] = 1.0 * ksds1.good.cumsum()/sum(ksds1.good)
  ksds1['cumsum_bad1'] = 1.0 * ksds1.bad.cumsum()/sum(ksds1.bad)
  
  
  if asc==1:
    ksds2 = ksds.sort_values(by=['pred','bad'],ascending=[True,False])
  elif asc==0:
    ksds2 = ksds.sort_values(by=['pred','bad'],ascending=[False,False])
  
  ksds2.index=range(len(ksds2.pred))
  ksds2['cumsum_good2'] = 1.0 * ksds2.good.cumsum()/sum(ksds2.good)
  ksds2['cumsum_bad2'] = 1.0 * ksds2.bad.cumsum()/sum(ksds2.bad)
  
  ksds = ksds1[['cumsum_good1','cumsum_bad1']]
  ksdsc = ksds.copy() # 创建副本并只在副本上操作
  ksdsc['cumsum_good2']=ksds2['cumsum_good2']
  ksdsc['cumsum_bad2'] = ksds2['cumsum_bad2']
  ksdsc['cumsum_good'] =(ksdsc['cumsum_good2']+ksdsc['cumsum_good1'])/2 # 累计负样本率(风险标签0)
  ksdsc['cumsum_bad'] = (ksdsc['cumsum_bad2']+ksdsc['cumsum_bad1'])/2   # 累计正样本率(风险标签1)
  
  
  #ks
  ksdsc['ks'] = ksdsc['cumsum_bad']-ksdsc['cumsum_good']
  ksdsc['tile0'] =range(1,len(ksdsc.ks)+1)
  ksdsc['tile'] = 1.0*ksdsc['tile0']/len(ksdsc['tile0'])
  
  
  qe=list(np.arange(0,1,1.0/n))
  qe.append(1)
  qe=qe[1:]
  
  ks_index = pd.Series(ksdsc.index)
  ks_index = ks_index.quantile(q=qe)
  ks_index = np.ceil(ks_index).astype(int)
  ks_index = list(ks_index)
  
  
  ksdsc = ksdsc.loc[ks_index]
  ksdsc = ksdsc[['tile','cumsum_good','cumsum_bad','ks']]
  ksds0 = np.array([[0,0,0,0]])
  ksdsc = np.concatenate([ksds0,ksdsc],axis=0)
  ksdsc = pd.DataFrame(ksdsc,columns=['tile','cumsum_good','cumsum_bad','ks'])

  #lift
  ksdsc['lift'] = ksdsc['cumsum_bad']/ksdsc['tile']
  
  ks_value =ksdsc.ks.max()
  ks_pop = ksdsc.tile[ksdsc.ks.idxmax()]
  print('ks_value is '+str(np.round(ks_value,4))+'at pop = '+str(np.round(ks_pop,4)))
  
  
  #chart
  
  plt.plot(ksdsc.tile,ksdsc.cumsum_good,label='cum_good',
          color='blue',linestyle='-',linewidth=2)
  plt.plot(ksdsc.tile,ksdsc.cumsum_bad,label='cum_bad',
          color='red',linestyle='-',linewidth=2)
  plt.plot(ksdsc.tile,ksdsc.ks,label='ks',
          color='green',linestyle='-',linewidth=2)
  
  plt.axvline(ks_pop,color='gray',linestyle='--')
  plt.axhline(ks_value,color='green',linestyle='--')
  plt.axhline(ksdsc.loc[ksdsc.ks.idxmax(),'cumsum_good'],color='blue',linestyle='--')
  plt.axhline(ksdsc.loc[ksdsc.ks.idxmax(),'cumsum_bad'],color='red',linestyle='--')
  
  plt.title('KS=%s ' %np.round(ks_value,4) +
           'at POP=%s'%np.round(ks_pop,4),fontsize=15 )
  plt.show()

  return ksdsc, plt

