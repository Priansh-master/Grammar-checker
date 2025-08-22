import pickle
import numpy as np
import os
from flask import Flask, render_template, request, jsonify

# Add this after the existing imports and before loading training data

# Read the same training file used for training
with open("smalltrial.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Get words array from text
words = text.split()

# Load training data
with open('training_data.pkl', 'rb') as f:
    training_data = pickle.load(f)

# Extract variables
vocab = training_data['vocab']
vocab_size = training_data['vocab_size'] 
embed_dim = training_data['embed_dim']
context_size = training_data['context_size']
word_to_ix = training_data['word_to_ix']
ix_to_word = training_data['ix_to_word']

# Load weights
embeddings = np.load('embeddings.npy')
theta = np.load('theta.npy')

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


def linear(m, theta):
    w = theta
    return m.dot(w)

def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())

def optimize(theta, grad, lr=0.01):
    theta -= grad * lr
    return theta

def forward(context_idxs, theta):
    m = embeddings[context_idxs].reshape(1, -1)
    n = linear(m, theta)
    o = log_softmax(n)
    
    return m, n, o

def predict(words):
    context_idxs = np.array([word_to_ix[w] for w in words])
    preds = forward(context_idxs, theta)
    word = ix_to_word[np.argmax(preds[-1])]
    
    return word

from numpy import loadtxt
# load array
datawtfile = loadtxt('data.csv')
#datawtfile = loadtxt('data.csv', delimiter=',')
# print the array
#print(datawtfile)
theta = optimize(theta, datawtfile, lr=0.01)


app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/check-grammar', methods=['POST'])
def check_grammar():
    # Get input from frontend
    user_input = request.json.get('sentence', '')
    
    # Write to output.txt
    with open('output.txt', 'w', encoding="utf-8") as file:
        file.write(user_input)
    

    sentences = user_input
    originalsent1 = user_input
    neworiginal = originalsent1.split()
    #print("original sent =",originalsent1)
    neworiginal = originalsent1.split()
    #print("new original sent =",neworiginal)

    ct = 0
    for l in range(len(neworiginal)):
        
        if l != 0:
            if ct == 0:
                newotail = neworiginal[1]
                ct = 1
            else:
                newotail = newotail +" "+neworiginal[l]
    #print("new orinial again =",newotail)      
        
    #sentences = sentences.lower()
    newwords = sentences.split()
    if len(newwords) < 6:
        sentences = sentences +" "+sentences
    #print(sentences)

    newwords = sentences.split()
    if len(newwords) < 6:
        sentences = sentences +" "+sentences
    #print(sentences)

    if len(newwords) < 6:
        sentences = sentences +" "+sentences
    #print(sentences)

    newwords = sentences.split()
    vocab = set(newwords)

    #print("vocab =",vocab)
    stopwords = ['|','।']
    newname = getwords(newwords,stopwords)
    #print("main new words =",newwords)

    contextwords = [0]*len(newname)
    contextname = [0]*len(newname)
    #print("len of newname =",len(newname))
    lennewname = len(newname)



    for i in range(0, len(newname)-4,1 ):
        #print("check word ",i," =",newname[i])
        allcontextname = [newname[i-2],newname[i-1],newname[i],newname[i+1], newname[i+2]]
        #print("all context test words =",allcontextname)
        #acceptance = getcontext(i,words[i], words[i-2],words[i-1],words[i+1], words[i+2],stopwords)
        #print("accept1 =",acceptance)
        #if acceptance == "true":
        contextname = [newname[i-1],newname[i],newname[i+1], newname[i+2]]
        targetname = newname[i-2]
        #data.append((contextname, targetname))
        #print(contextname, newname)
        #print("2 context words at ",i," =",contextname)
        

    #for i in range(0, len(newname)-2,1 ):
 #   print("check word ",i," =",newname[i])
  #  allcontextname = [newname[i],newname[i+1]]
#    print("all context test words =",allcontextname)
    #acceptance = getcontext(i,words[i], words[i-2],words[i-1],words[i+1], words[i+2],stopwords)
    #print("accept1 =",acceptance)
    #if acceptance == "true":
 #   contextname = [newname[i],newname[i+1]]
  #  targetname = newname[i+2]
    #data.append((contextname, targetname))
   # print(contextname, newname)
   # print("3 context words at ",i," =",contextname)
   # print(predict(contextname))
    
    testcontaxt = contextname
    #print("final contaxt word=",testcontaxt)
    testcontaxt = [newname[-1],newname[0],newname[1]]
    #print("testcontaxt=",testcontaxt)
    #predict(testcontaxt)
    difference = 0
    for i in range(len(testcontaxt)):
        agreement = 0
        for j in range(len(words)):
            if testcontaxt[i] == words[j]:
                #print("good testcontaxt[",i,"]=",testcontaxt[i])
                good = 1
                agreement += 1
                #print("agreement =",agreement)
                break
            else:
                good = 0
                agreement =0
                testcontaxt[i] = newname[i-2]
                #print("bad testcontaxt[",j,"]=",testcontaxt[j])
                
                #break
                
        if agreement < 1:
            testcontaxt[i] = words[0]
            difference += 1
    #print("difference =",difference)            
            
    #print("new textcontaxt =",testcontaxt)      
    #predict(testcontaxt)

    contaxt1 = [testcontaxt[-1],testcontaxt[0],testcontaxt[1],testcontaxt[2]]
    contaxt2 = [testcontaxt[-2],testcontaxt[-1],testcontaxt[0],testcontaxt[1]]
    #contaxt3 = [testcontaxt[-1],testcontaxt[1],testcontaxt[-1],testcontaxt[1]]
    #contaxt1 = [testcontaxt[-1],testcontaxt[1],testcontaxt[-1],testcontaxt[1]]

    pred = [0]*4
    #testcontaxt1 = [testcontaxt[-1],testcontaxt[1],testcontaxt[-1],testcontaxt[1]]
    testcontaxt1 = [testcontaxt[-1], testcontaxt[0]]
    #print("1 testcontaxt1=",testcontaxt1)
    #print(predict(testcontaxt1))
    pred[0] = predict(testcontaxt1)
    print("Correct sentence is ",testcontaxt[-1], testcontaxt[0],pred[0])

    #testcontaxt2 = [testcontaxt[1],testcontaxt[2], testcontaxt[1],testcontaxt[2]]
    testcontaxt2 = [testcontaxt[0],testcontaxt[1]]
    #print("2 testcontaxt2=",testcontaxt2)
    #print(predict(testcontaxt2))
    pred[1] = predict(testcontaxt2)
    print("Correct sentence is ",pred[1],testcontaxt[0],testcontaxt[1])

    #testcontaxt3 = [testcontaxt[3],testcontaxt[2],testcontaxt[3],testcontaxt[2]]
    #print("3 testcontaxt3=",testcontaxt3)
    #print(predict(testcontaxt3))
    #pred[2] = predict(testcontaxt3)

    #testcontaxt4 = [testcontaxt[2],testcontaxt[3],testcontaxt[2],testcontaxt[3]]
    #print("4 testcontaxt4=",testcontaxt4)
    #print(predict(testcontaxt4))
    #pred[3] = predict(testcontaxt4)



    pre = 0
    pos = "hi"
    full = "|"
    for i in range(len(pred)):
        ct = 0
        #print("i =",pred[i])
        for j in range(len(newwords)):
            #print("j =",newwords[j],"full =",full)
            if pred[i] != newwords[j] :
                if pred[i] != full:
                    pre = 1
                    ct+=1
                    pos = pred[i]
                    if i == 0:
                        #print("Correct pred 0 =",pred[i],testcontaxt1)
                        for s in range(len(testcontaxt1)):
                            if s == 0:
                                correctsen1 = testcontaxt1[s]
                            else:
                                correctsen1 = correctsen1 +" "+testcontaxt1[s]
                            #print("Correct pred00 =",newotail,pred[i])
                    else:
                        #print("Correct pred 1 =",pred[i],testcontaxt2)
                        for s in range(len(testcontaxt2)):
                            if s == 0:
                                correctsen2 = testcontaxt2[s]
                            else:
                                correctsen2 = correctsen2 +" "+testcontaxt2[s]
                            #print("Correct pred11 =",pred[i],newotail)
            else:
                pre = 0
                break
        #print("pre =",pre, "ct =",ct)
        if ct == len(newwords):
            print("prediction =",pos)
            
    #print("LATER Correct sentence is ",pred[0],contaxt1)
    
    # Collect results
    results = []
    if pred[0]:
        results.append(f"Correct sentence is {testcontaxt[-1]} {testcontaxt[0]} {pred[0]}")
    if pred[1]:
        results.append(f"Correct sentence is {pred[1]} {testcontaxt[0]} {testcontaxt[1]}")
    
    return jsonify({
        'original': user_input,
        'corrections': results
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render gives $PORT
    app.run(host="0.0.0.0", port=port, debug=True)
