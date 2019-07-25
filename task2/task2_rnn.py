import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils import data

use_cuda=torch.cuda.is_available()
device=torch.device('cuda:0' if use_cuda else "cpu")

Epoach=100
num_class=5
sentence_maxlen=50
Embedding_size=100
Hidden_size=256
dropout_rate=0.5
embed_dropout=0.5
Lr=0.001
wd=0.06
stop_words=['.',',',';','(',')','[',']','=','?','{','}']

def load_data(path):
    file=pd.read_csv(path,sep='\t',header=0,index='PhraseId')
    file=np.array(file)
    num=file.shape[0]
    for i in range(num):
        file[i][1]=file[i][1].lower()
    return file,num

def read_pretrain(path):
    emb=[]
    vocab=[]
    dic={}
    index=0
    with open(path,'r',encoding='UTF-8') as f:
        for line in f.readlines():
            row=line.strip().split()
            emb.append(row[1:])
            vocab.append(row[0])
            dic[row[0]]=index
            index+=1
    return vocab,emb,dic

def read_stop_word(path):
    with open(path) as f:
        x=''
        for char in f.readlines():
            x=x+char
        x=x.split('\n')
    return x

class CustomDataset(data.Dataset):
    def __init__(self,datas,labels):
        self.datas=datas
        self.labels=labels
    def __getitem__(self, index):
        sentence,label=self.datas[index],self.labels[index]
        return sentence,label
    def __len__(self):
        return len(self.datas)

def make_dicts(text):
    dicts=set()
    for sentence in text:
        sentence_list=sentence.split()
        for word in sentence_list:
            dicts.add(word)
    return dicts

def one_hot(value,num):
    out=np.zeros()
    out[value]=1
    return out

def make_data(dataset,train=True):
    num=dataset.shape[0]
    out=[]
    label=[]
    for i in range(num):
        sentence=dataset[i][1]
        embed=[]
        sentence_list=sentence.split()
        for word in sentence_list:
            if word in total_dict:
                embed.append(dicts[word])
            else:
                continue
        if len(embed)<=sentence_maxlen:
            for j in range(sentence_maxlen-len(embed)):
                embed.append(0)
        else:
            embed=embed[:sentence_maxlen]
        out.append(embed)
        if train==True:
            label.append(dataset[i][2])
    return out,label

def softmax(y_hat):
    num=y_hat.shape[1]
    y_ave=np.sum(y_hat,axis=1)/num
    y_hat=(y_hat.T-y_ave).T
    exp_y=np.sum(np.exp(y_hat),axis=1)
    softmax_y=(np.exp(y_hat.T))/exp_y
    softmax_y=softmax_y.T
    return softmax_y

# total_num 总的单词数目
class RNN(nn.Module):
    def __init__(self,total_num,embed_size,hidden_size,layers,dropout,embed_dropout_,weight,numclass):
        super(RNN, self).__init__()
        self.total_num=total_num
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.layers=layers
        self.dropout=dropout
        self.embed_dropout_=embed_dropout_
        self.weight=weight
        self.num_class=num_class

        self.embedding=nn.Embedding(self.total_num,self.embed_size)
        self.embedding.weight.data.copy_(self.weight)
        self.embed_dropout=nn.Dropout(self.embed_dropout_)
        self.lstm = nn.LSTM(self.embed_size,self.hidden_size,self.layers,batch_first=True,dropout=self.dropout)
        self.dropout_layer=nn.Dropout(self.dropout)
        self,fc=nn.Linear(self.hidden_size,self.num_class)
        '''  
        LSTM:
        input_size：x的特征维度
        hidden_size：隐藏层的特征维度
        num_layers：lstm隐层的层数，默认为1
        bias：False则bihbih = 0
        默认为True
        batch_first：True则输入输出的数据格式为(batch, seq, feature)
        dropout：除最后一层，每一层的输出都进行dropout，默认为: 0
        bidirectional：True则为双向lstm默认为False
        输入：input, (h0, c0)
        输出：output, (hn, cn)
        '''

    def foward(self,x,intial=None):
        # x.shape=batch,seqlen
        x=self.embedding(x)
        x=self.embed_dropout(x)
        if initial==None:
            h0=torch.zeros(self.layers,x.shape[0],self.hidden_size).to(device)
            c0=torch.zeros(self.layers,x.shape[0],self.hidden_size).to(device)
        else:
            (h0,c0)=intial
        out,(hn,cn)=self.lstm(x,(h0,c0))
        out=self.dropout_layer(out)
        out=torch.tanh(out)
        #out.shape=batch,seqlen,hidden_size
        result=self.fc(out[:,-1,:])
        #result.shape=batch,numclass
        return result

#train
train_data,num_train=load_data('../task1/train.tsv')
test_data,num_test=load_data('../task1/test.tsv')
glove_vocab,glove_emb,dic=read_pretrain('glove_6B_100d.txt')

stop_english=read_stop_word('stop_word_english.txt')
stop_words.extend(stop_english)
train_sentence=[]
test_sentence=[]
for i in range(num_train):
    train_sentence.append(train_data[i][1])
for i in range(num_test):
    test_sentence.append(test_data[i][1])
train_dict=make_dicts(train_sentence)
test_dict=make_dicts(test_sentence)
total_dict=train_dict|test_dict
total_dict=sorted(list(total_dict))
total_num_word=len(total_dict)
dicts={w:i for i,w in enumerate(total_dict)}

pretrain_weight=np.zeros([total_num_word,Embedding_size])
for i in range(total_num_word):
    word=total_dict[i]
    if word in glove_vocab:
        num=dic[word]
        vector=glove_emb[num]
        vector=np.array(vector)
        pretrain_weight[i]=vector
    else:
        pretrain_weight[i]=np.random.rand(1,Embedding_size)

train_emb,train_label=make_data(train_data,train=True)
test_emb,_=make_data(test_data,train=False)
train_emb=np.array(train_emb)
train_label=np.array(train_label)
test_emb=np.array(test_emb)

train_set=CustomDataset(train_emb,train_label)
train_generator=data.DataLoader(train_set,**params)

weight=torch.FloatTensor(pretrain_weight)
net=RNN(total_num_word,Embedding_size,Hidden_size,2,dropout_rate,embed_dropout,weight,numclass=num_class).to(device)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters())

net.train(model=True)
for epoch in range(Epoach):
    for i,(batch_train,batch_label) in enumerate(train_generator):
        batch_train=batch_train.type(torch.long).to(device)
        batch_label=batch_label.type(torch.long).to(device)
        out=net(batch_train)
        loss=criterion(out,batch_label)
        if i%20==0:
            print('Loss:{:.3f}'.format(loss.item()))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(net.parameters(),5) #防止梯度爆炸
        optimizer.step()
        del batch_train,batch_label,loss,out

#test
net.eval()
test_emb=torch.from_numpy(test_emb).long()
test_emb=test_emb.to(device)
result=net(test_emb)
result=result.to(device)
result=result.detach().numpy()
result=softmax(result)
result_=np.argmax(result,axis=1)
result_=list(result_)
num_list=list(range(15601,15601+num_test))
dataframe=pd.DataFrame({'PhraseId':num_list,'Sentiment':result_})
dataframe.to_csv('result.csv',index=False,sep=',')





