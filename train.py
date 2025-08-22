def getwords(wods,stop):
    #print("wods =",wods,"  stop=",stop)
    #print("length of words =",len(wods),"  length of stops =",len(stop))
   # unicodesent = wds
    #unicodestop = stp
    #for i in range(0,4,1):
    unicodenewsent = [0]*len(wods)
    for i in range(len(wods)):
        unicodenewsent[i] = wods[i].encode("utf-8")
        #print("unicodenewsent[",i,"]=",unicodenewsent[i],"for the word[",i,"]=",wods[i])

        #print("stop=",stop)
    #unicodestop = stopwords
    unicodenewstop = [0]*len(stop)
    for i in range(len(stop)):
        unicodenewstop[i] = stop[i].encode("utf-8")
       # print("unicodenewstop[",i,"]=",unicodenewstop[i],"for the stopword[",i,"]=",stop[i])

   # print("words[",(0),"]=",wods[0])
    newwds = []
   #  print("words[",(5),"]=",wds[5])
   # print("len of stop =",len(stop))
    for i in range(len(wods)):
      #  print("index for loop i = ",i,'word =',wods[i])
        count = 0
        for j in range(len(stop)):
          #  print("j=",j)
           # print(stop[j])
            if unicodenewsent[i] == unicodenewstop[j]:
                mainnindex = 1
                #print("success")
               # return "wrong"
                #return ["n","n","n","n"]
                #stopindex = i
                #break
                #return 0
                #i += 1
                count = -100
            else:
                mainindex = 0
                count+=1
                #print("words[",(i),"]=",wods[i],"  count=",count)
                #print("unicodenewsen[",(i),"]=",unicodenewsent[i]," index words[",(i),"]=",wods[i])
                #print("stopnewwords[",j,"]=",stop[j])
                #print("stop unicodenewstop[",j,"]=",unicodenewstop[j],"  stop index words[",j,"]=",stop[j])
        if count == len(stop):
            #print("semi final list for word ",'i=',wods[i],' new words =',newwds)
            newwds.append(wods[i])
               # return correct
            
     
            
        
            
    #print("main index =",mainindex)
    #print("new final list =",newwds)
    return newwds


# Weight file generating
# NLP
import codecs
from itertools import permutations
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,\
    Embedding, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
#import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
# define data
datafile = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# save to csv file
savetxt('data.csv', datafile, delimiter=',')


#file = open("E://pyt//nepali-english//Royal_data.txt","r")
#file = codecs.open("E://pyt//nepali-english//Royal_data.txt", "r","utf-8")
#file1 = file
#royal_data = file.readlines()
#royal_data1 = file.read()
#print(royal_data)
#file.close()

#with open("E://pyt//nepali-english//grammarcheck.txt", "r", encoding="utf-8") as f:
with open("smalltrial.txt", "r", encoding="utf-8") as f:
    text = f.read()
    print(text)



#finalroyal_data = []
#for i in range(len(royal_data)):
#    royal_data[i] = royal_data[i].lower().replace('\n', ' ')
 #   finalroyal_data.append(royal_data[i])

#print(royal_data)   
stopwords = ['the', 'is', 'will', 'be', 'a', 'only', 'can', 'their', 'now', 'and', 'at', 'it','में', 'सा', 'होता', 'है', 'है?', 'था?', 'था', 'का', 'के', 'कौन', 'पर', 'जाता', 'कब', 'कौन-सा', 'प्राप्त', 'हुआ','?','will','be']
stopwords = ['में','.', ':-','के','की','?','का','होता', 'कौन', 'सा','को','है?', 'है', 'था?','था','ने','थे?','थे','a','A','be','only','now', 'will','|','।']
#filtered_data = []
#for sent in royal_data:
 #   temp = []
  #  for word in sent.split():
   #     if word not in stopwords:
    #        temp.append(word)
    #filtered_data.append(temp)

#print(filtered_data)


#sentences = "The future king is the prince Daughter is the princess Son is the prince Only a man can be a king Only a woman can be a queen खट्टे फलों में कौन सा एसिड होता है ? साइट्रिक एसिड सम्राट अशोक किसका उत्तराधिकारी था ? बिंदुसार भारतीय संविधान को पहली बार कब संशोधित किया गया ? 1950 सिंधु घाटी सभ्यता का बन्दरगाह कहाँ पर था ? लोथल ISRO के हेड्क्वॉर्टर्स कहा है ? बैंगलोर भारत का सबसे छोटा राज्य कौन सा है ? गोवा The princess will be a queen The prince is a strong man"

# lower all characters
sentences = text
#sentences = sentences.lower()
words = sentences.split()
vocab = set(words)

print("vocab =",vocab)
vocab_size = len(vocab)
embed_dim = 140
context_size = 1

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

# data - [(context), target]
print("words =",words)

data = []

unicodesent1 = [0]*len(words)
unicodeend = "|".encode("utf-8")
print("unicodeend =",unicodeend)
for i in range(len(words)):
    unicodesent1[i] = words[i].encode("utf-8")
    #print("unicodesent[",i,"]=",unicodesent1[i],"for the word[",i,"]=",words[i])

print("stop=",stopwords)
unicodestop1 = [0]*len(words)
#unicodestop = stopwords
for i in range(len(stopwords)):
    unicodestop1[i] = stopwords[i].encode("utf-8")
    #print("unicodestop[",i,"]=",unicodestop1[i],"for the stopword[",i,"]=",stopwords[i])

#fourwords = ["a","b","c","d"]

print("123 words = ",words)
totalwords = []
cti = 0
twords = []
ttwords = []
for i in words:
    print("temp i =",i)
    print("unicode =",unicodesent1[cti])
    if unicodesent1[cti] != unicodeend:
        twords.append(i)
       # twords = []
        print("total twords =",twords)
    else:
        print("hi")
        ttwords.append(twords)
        print("ttwords =",ttwords)
        totalwords.append(twords)
        ttwords = []
        twords = []
        #ttwords.append(twords)
        #print("semi total words=",ttwords)
    cti = cti + 1
    


print("total words =",totalwords)

newwords = getwords(words,stopwords)
print("main new words =",newwords)

contextwords = [0]*len(newwords)
for i in range(0, len(newwords)-4,1 ):
    print("check word ",i," =",newwords[i])
    allcontextwords = [newwords[i-2],newwords[i-1],newwords[i],newwords[i+1], newwords[i+2]]
    print("4 all context words =",allcontextwords)
    #acceptance = getcontext(i,words[i], words[i-2],words[i-1],words[i+1], words[i+2],stopwords)
    #print("accept1 =",acceptance)
    #if acceptance == "true":
    contextwords = [newwords[i-1],newwords[i],newwords[i+1], newwords[i+2]]
    target = newwords[i-2]
    data.append((contextwords, target))
   # print(contextwords, newwords)
    #print("2 context words at ",i," =",contextwords)
        
    contextwords = [newwords[i-1],newwords[i],newwords[i+2], newwords[i+1]]
    target = newwords[i-2]
    data.append((contextwords, target))
   # print(contextwords, newwords)
    #print("3 context words at ",i," =",contextwords)
        
    contextwords = [newwords[i],newwords[i-1],newwords[i+2], newwords[i+1]]
    target = newwords[i-2]
    data.append((contextwords, target))
  #  print(contextwords, newwords)
   # print("4 context words at ",i," =",contextwords)
        
    contextwords = [newwords[i],newwords[i-1],newwords[i+2], newwords[i+1]]
    target = newwords[i-2]
    data.append((contextwords, target))
    print(contextwords, newwords)
    print("5 context words at ",i," =",contextwords)
        
    contextwords = [newwords[i+1],newwords[i+2],newwords[i], newwords[i-1]]
    target = newwords[i-2]
    data.append((contextwords, target))
    print(contextwords, newwords)
    print("6 context words at ",i," =",contextwords)
        
    contextwords = [newwords[i-1],newwords[i],newwords[i+1], newwords[i+2]]
    target = newwords[i-2]
    data.append((contextwords, target))
  #  print(contextwords, newwords)
   # print("7 context words at ",i," =",contextwords)
    
    contextwords = [newwords[i],newwords[i-1],newwords[i-2], newwords[i+1]]
    target = newwords[i+2]
    data.append((contextwords, target))
#    print(contextwords, newwords)
 #   print("8 context words at ",i," =",contextwords)
    
    contextwords = [newwords[i-1],newwords[i],newwords[i-2], newwords[i+1]]
    target = newwords[i+2]
    data.append((contextwords, target))
 #   print(contextwords, newwords)
  #  print("9 context words at ",i," =",contextwords)
    
    contextwords = [newwords[i-2],newwords[i-1],newwords[i], newwords[i+1]]
    target = newwords[i+2]
    data.append((contextwords, target))
#    print(contextwords, newwords)
 #   print("10 context words at ",i," =",contextwords)
    
    contextwords = [newwords[i-1],newwords[i-2],newwords[i], newwords[i+1]]
    target = newwords[i+2]
    data.append((contextwords, target))
 #   print(contextwords, newwords)
  #  print("10 context words at ",i," =",contextwords)
    
    contextwords = [newwords[i+1],newwords[i],newwords[i-1], newwords[i-2]]
    target = newwords[i+2]
    data.append((contextwords, target))
    print(contextwords, newwords)
    print("11 context words at ",i," =",contextwords)
    
    contextwords = [newwords[i],newwords[i+2],newwords[i-1], newwords[i-2]]
    target = newwords[i+1]
    data.append((contextwords, target))
 #   print(contextwords, newwords)
   # print("12 context words at ",i," =",contextwords)
    
    contextwords = [newwords[i+2],newwords[i],newwords[i-1], newwords[i-2]]
    target = newwords[i+1]
    data.append((contextwords, target))
 #   print(contextwords, newwords)
  #  print("13 context words at ",i," =",contextwords)
        
 


#print(data[:5])

print("\n Data =",data)
print("context words 4 length =",data)

# 3 inputs small sentences
#print(data[:5])
data1 = []

contextwords1 = [0]*len(newwords)
for i in range(0, len(newwords)-3,1 ):
    print("check word ",i," =",newwords[i])
    allcontextwords1 = [newwords[i-2],newwords[i-1],newwords[i],newwords[i+1],newwords[i+2]]
    print("3 all context words =",allcontextwords1)
    #acceptance = getcontext(i,words[i], words[i-2],words[i-1],words[i+1], words[i+2],stopwords)
    #print("accept1 =",acceptance)
    #if acceptance == "true":
    contextwords1 = [newwords[i-1],newwords[i],newwords[i+1]]
    target = newwords[i-2]
    data1.append((contextwords1, target))
   # print(contextwords, newwords)
    #print("2 context words at ",i," =",contextwords)
        
    contextwords1 = [newwords[i-1],newwords[i],newwords[i+2]]
    target = newwords[i-2]
    data1.append((contextwords1, target))
   # print(contextwords, newwords)
    #print("3 context words at ",i," =",contextwords)
        
    contextwords1 = [newwords[i],newwords[i-1],newwords[i+2]]
    target = newwords[i-2]
    data1.append((contextwords1, target))
  #  print(contextwords, newwords)
   # print("4 context words at ",i," =",contextwords)
        
    contextwords1 = [newwords[i],newwords[i-1],newwords[i+2]]
    target = newwords[i-2]
    data1.append((contextwords1, target))
    print(contextwords1, newwords)
    print("5 context words at ",i," =",contextwords1)
        
    contextwords1 = [newwords[i+1],newwords[i+2],newwords[i]]
    target = newwords[i-2]
    data1.append((contextwords1, target))
    print(contextwords1, newwords)
    print("6 context words at ",i," =",contextwords1)
        
    contextwords1 = [newwords[i-1],newwords[i],newwords[i+1]]
    target = newwords[i-2]
    data1.append((contextwords1, target))
  #  print(contextwords, newwords)
   # print("7 context words at ",i," =",contextwords)
    
    contextwords1 = [newwords[i],newwords[i-1],newwords[i-2]]
    target = newwords[i+2]
    data1.append((contextwords1, target))
#    print(contextwords, newwords)
 #   print("8 context words at ",i," =",contextwords)
    
    contextwords1 = [newwords[i-1],newwords[i],newwords[i-2]]
    target = newwords[i+2]
    data1.append((contextwords1, target))
 #   print(contextwords, newwords)
  #  print("9 context words at ",i," =",contextwords)
    
    contextwords1 = [newwords[i-2],newwords[i-1],newwords[i]]
    target = newwords[i+2]
    data1.append((contextwords1, target))
#    print(contextwords, newwords)
 #   print("10 context words at ",i," =",contextwords)
    
    contextwords1 = [newwords[i-1],newwords[i-2],newwords[i]]
    target = newwords[i+2]
    data1.append((contextwords1, target))
 #   print(contextwords, newwords)
  #  print("10 context words at ",i," =",contextwords)
    
    contextwords1 = [newwords[i+1],newwords[i],newwords[i-1]]
    target = newwords[i+2]
    data1.append((contextwords1, target))
    print(contextwords1, newwords)
    print("11 context words at ",i," =",contextwords)
    
    contextwords1 = [newwords[i],newwords[i+2],newwords[i-1]]
    target = newwords[i+1]
    data1.append((contextwords1, target))
 #   print(contextwords, newwords)
   # print("12 context words at ",i," =",contextwords)
    
    contextwords1 = [newwords[i+2],newwords[i],newwords[i-1]]
    target = newwords[i+1]
    data1.append((contextwords1, target))
 #   print(contextwords, newwords)
  #  print("13 context words at ",i," =",contextwords)
        
 



contextwords2 = [0]*len(newwords)
data2 = []
for i in range(0, len(newwords)-2,1 ):
    print("check word ",i," =",newwords[i])
    allcontextwords1 = [newwords[i-2],newwords[i-1],newwords[i],newwords[i+1],newwords[i+2]]
    print("2 all context words =",allcontextwords1)
    #acceptance = getcontext(i,words[i], words[i-2],words[i-1],words[i+1], words[i+2],stopwords)
    #print("accept1 =",acceptance)
    #if acceptance == "true":
    contextwords2 = [newwords[i-1],newwords[i]]
    target = newwords[i+1]
    data2.append((contextwords2, target))
   # print(contextwords, newwords)
    #print("2 context words at ",i," =",contextwords)
        
    #contextwords2 = [newwords[i],newwords[i+1]]
    #target2 = newwords[i+2]
    #data2.append((contextwords2, target2))
    
    contextwords2 = [newwords[i],newwords[i+1]]
    target2 = newwords[i-1]
    data2.append((contextwords2, target2))
    
    #contextwords2 = [newwords[i+1],newwords[i+2]]
    #target2 = newwords[i]
    #data2.append((contextwords2, target2))



print("\n Data1 =",data2)
print("context words 2 length =",data2)

data3 = []
for a in range(0,len(totalwords),1):
    print("\n total words =",totalwords)
    sentwd = totalwords[a]
    print("total check sent ",a," =",sentwd)
    for b in range(0,len(sentwd)-1,1):
        print("total check sent inside",a," =",totalwords[a])
        print("total word=",sentwd[b])
        contextwords3 = [sentwd[b-1],sentwd[b]]
        target = sentwd[b+1]
        data3.append((contextwords3, target))
   # print(contextwords, newwords)
    #print("2 context words at ",i," =",contextwords)
        
    #contextwords2 = [newwords[i],newwords[i+1]]
    #target2 = newwords[i+2]
    #data2.append((contextwords2, target2))
    
        contextwords3 = [sentwd[b],sentwd[b+1]]
        target = sentwd[b-1]
        data3.append((contextwords3, target))
        
print("total data3 =",data3)
data = data3


print("\n Data3 =",data3)
print("Total context words 2 length =",data3)


#data = data1
print("\n Data =",data)
print("context words =",data)
data = data3
print("final context words =",data)
embeddings =  np.random.random_sample((vocab_size, embed_dim))

def linear(m, theta):
    w = theta
    return m.dot(w)

def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())

def NLLLoss(logs, targets):
    out = logs[range(len(targets)), targets]
    return -out.sum()/len(out)

def log_softmax_crossentropy_with_logits(logits,target):

    out = np.zeros_like(logits)
    out[np.arange(len(logits)),target] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    
    return (- out + softmax) / logits.shape[0]

def forward(context_idxs, theta):
    m = embeddings[context_idxs].reshape(1, -1)
    n = linear(m, theta)
    o = log_softmax(n)
    
    return m, n, o

def backward(preds, theta, target_idxs):
    m, n, o = preds
    
    dlog = log_softmax_crossentropy_with_logits(n, target_idxs)
    dw = m.T.dot(dlog)
    
    return dw

def optimize(theta, grad, lr=0.01):
    theta -= grad * lr
    return theta

theta = np.random.uniform(-1, 1, (2 * context_size * embed_dim, vocab_size))

epoch_losses = {}

for epoch in range(180):

    losses =  []
    #for contextwords2, target2 in data2:
    for contextwords, target in data:
        context_idxs = np.array([word_to_ix[w] for w in contextwords])
        preds = forward(context_idxs, theta)

        target_idxs = np.array([word_to_ix[target]])
        loss = NLLLoss(preds[-1], target_idxs)

        losses.append(loss)

        grad = backward(preds, theta, target_idxs)
        #datafile = grad
        #savetxt('E:/pyt/nepali-english/data.csv', datafile, delimiter=',')
        #print("grad = ",grad)
        theta = optimize(theta, grad, lr=0.01)
        
     
    epoch_losses[epoch] = losses
    

ix = np.arange(0,80)

fig = plt.figure()
fig.suptitle('Epoch/Losses', fontsize=20)
plt.plot(ix,[epoch_losses[i][0] for i in ix])
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Losses', fontsize=12)
#print("Losses=",epoch_losses[ix][0])
plt.grid()

def predict(words):
    # Check if all words are in vocabulary 
    for w in words:
        if w not in word_to_ix:
            print(f"Warning: '{w}' not found in vocabulary")
            return None # Return None for unknown words
            
    context_idxs = np.array([word_to_ix[w] for w in words])
    preds = forward(context_idxs, theta)
    word = ix_to_word[np.argmax(preds[-1])]
    print("prediction word =", word)
    return word

# Test with error handling
result = predict(['बैंगलोर', 'भारत', 'छोटा', 'राज्य'])
if result is None:
    print("Could not make prediction - some words are unknown")


def accuracy():
    wrong = 0

    for context, target in data:
        if(predict(context) != target):
            wrong += 1
            
    return (1 - (wrong / len(data)))

accuracy()


from numpy import asarray
from numpy import savetxt
# define data
datafile = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# save to csv file
savetxt('data.csv', datafile)
datafile = grad
savetxt('data.csv', datafile)
print("grad = ",grad)



import pickle
import numpy as np

# Save vocabulary and embeddings
training_data = {
    'vocab': vocab,
    'vocab_size': vocab_size,
    'embed_dim': embed_dim,
    'context_size': context_size,
    'word_to_ix': word_to_ix,
    'ix_to_word': ix_to_word,
    'embeddings': embeddings,
    'theta': theta
}

# Save to binary file
with open('training_data.pkl', 'wb') as f:
    pickle.dump(training_data, f)

# Save weights separately as numpy array
np.save('embeddings.npy', embeddings)
np.save('theta.npy', theta)