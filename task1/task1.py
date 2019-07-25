import numpy as np
import pandas as pd

num_class=5
Epoch=500
Batch_size=64
Lr=0.003
n_gram=2
wd=0.08

def load_data(path):
    file=pd.read_csv(path,sep='\t',header=0,index_col='PhraseId')
    file=np.array(file)
    num=file.shape[0]
    #print(file)
    for i in range(num):
        file[i][1]=file[i][1].lower()
    return file,num

def make_n_dicts(sentences,n=2):
    dicts=set()
    dicts.add('')
    for sentence in sentences:
        sentence_list=sentence.split()
        for i in range(len(sentence_list)):
            for j in range(n):
                if i-j>=0:
                    dicts.add(''.join([word for word in sentence_list[i-j:i+1]]))
    dicts=sorted(list(dicts))
    return dicts

def one_hot(number,num_classes):
    out=np.zeros([num_classes])
    out[number]=1
    return out

def one_hots(y,num_classes):
    out=[]
    for i in range(y.shape[0]):
        out.append(one_hot(y[i],num_classes))
    out=np.array(out)
    return out

def softmax(y_hat):
    num=y_hat.shape[1]
    y_ave=np.sum(y_hat,axis=1)/num
    #print(y_ave)
    y_hat=(y_hat.T-y_ave).T
    exp_y=np.sum(np.exp(y_hat),axis=1)
    softmax_y=(np.exp(y_hat.T))/exp_y
    softmax_y=softmax_y.T
    return  softmax_y

def find_max_index(softmax_y):
    return np.argmax(softmax_y,axis=1)

def compute_accuracy(y1,y):
    num=y1.shape[0]
    z1=np.argmax(y1,axis=1)
    z=np.argmax(y,axis=1)
    return np.sum(z1==z)/num

def comput_accuracy_loss(dataset,labels,W):
    out=dataset.dot(W)
    softmax_y=softmax(out)
    y1=find_max_index(softmax_y)
    y1=one_hots(y1,num_class)
    accuracy=compute_accuracy(y1,labels)
    m=labels.shape[0]
    loss=-np.sum((np.log(softmax_y)/m)*labels)
    return y1,loss,accuracy

def L2(x):
    return np.sum(x**2)/2

def make_dataset(X,dicts,train=True):
    dataset=[]
    labels=[]
    num_x=X.shape[0]
    for i in range(num_x):
        sentence=X[i][1]
        feature=np.array([word in sentence for word in dicts],dtype=np.int16)
        dataset.append(feature)
        if train:
            labels.append(one_hot(X[i][2],num_class))
    dataset=np.array(dataset)
    labels=np.array(labels)
    return dataset,labels

def train(dataset,labels,batch_size,EPOCH,lr):
    n=dataset.shape[0]
    m=dataset.shape[1]
    total=0
    average_acc=0
    W = 0.1 * np.random.randn(m, num_class)
    W[0, :] = np.zeros(num_class)
    for epoch in range(EPOCH):
        data_and_lab=list(zip(dataset,labels))
        np.random.shuffle(data_and_lab)
        dataset,labels=zip(*data_and_lab)
        dataset=np.array(dataset)
        labels=np.array(labels)
        for i in range(n//batch_size+1):
            begin=i*batch_size
            end=(i+1)*batch_size
            if end>n:
                end=n
            mini_dataset=dataset[begin:end]
            mini_labels=labels[begin:end]
            y1,loss,_=comput_accuracy_loss(mini_dataset,mini_labels,W)
            dw=(mini_dataset.T).dot(y1-mini_labels)
            W=W-lr*dw

        if epoch%10==0:
            _,loss,accuracy=comput_accuracy_loss(dataset,labels,W)
            print('loss:{:.3f},accuracy:{:.3f}'.format(loss,100*accuracy))
            average_acc+=accuracy
            total+=1
    average_acc/=total
    return  W,average_acc

def prediction(dataset,W):
    out=dataset.dot(W)
    softmax_y=softmax(out)
    y1=find_max_index(softmax_y)
    return y1


train_data,train_num=load_data('train.tsv')

train_data=train_data[:7000]
num_train=train_data.shape[0]
sentences=[train_data[i][1] for i in range(num_train)]
dicts=make_n_dicts(sentences,n=n_gram)

train_dataset,train_labels=make_dataset(train_data,dicts,train=True)
W,average_acc=train(dataset=train_dataset,labels=train_labels,batch_size=Batch_size,EPOCH=Epoch,lr=Lr)
print('train_average_accuracy {:.3f}%'.format(100 * average_acc))

test_data,test_num=load_data('test.tsv')
test_dataset,_=make_dataset(test_data,dicts,train=False)
del test_data
result=prediction(test_dataset,W)
num_list=list(result)
dataframe=pd.DataFrame({'PhraseId':num_list, 'Sentiment':result})
dataframe.to_csv('training_num_{}_{}_gram_epoch_{}_batch_size_{}_lr_{:.3f}_wd_{:.3f}.csv'
                 .format(num_train, n_gram, Epoch, Batch_size, Lr, wd), index=False, sep=',')
#a=np.arange(15).reshape(3,5)
