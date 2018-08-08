# -*- coding: utf-8 -*-
"""
@author: Sanket Sheth (sas6792@g.rit.edu)
"""







#Step  1 and pre-processing:
from collections import OrderedDict
from operator import itemgetter

from math import log1p
    
import re
import nltk
global ans
global rulpr
global rul
global baseline

#Step 3 Measure Collocational Distribbutions

def coldist(dct,dctOT,dctB):
    d = OrderedDict(sorted(dct.items(), key=itemgetter(1)))
    l=list(d.items())[len(d)-2:]
    l=l[::-1]
    topL=[]
    for t in l:
        topL.append(t[0])
    FL={}
    SL={}
    
    for t in topL:
        if t in dctOT:
            FL[t]=dctOT[t]
        else:
            FL[t]=0
        if t in dctB:
            SL[t]=dctB[t]
        else:
            SL[t]=0
    return FL,SL

#Feature 1 Word -+K where k=1

def wordL(wordsleft,labels,i,k,check):
    for line in labels[i]:
        ind=line.index(check)
        ind=ind-k
        left=line[ind]
       # print(left)
        if left in wordsleft:
            wordsleft[left] += 1
        else:
            wordsleft[left]=1
    return wordsleft


def wordK(labels,k,check):
    wordsleftT={}
    wordsleftF={}   
    wordsleftS={}

    wordsleftT=wordL(wordsleftT,labels,0,k,check)
    wordsleftF=wordL(wordsleftF,labels,0,k,check)
    wordsleftT=wordL(wordsleftT,labels,1,k,check)
    wordsleftS=wordL(wordsleftS,labels,1,k,check)
    
    FL={}
    SL={}
    FL,SL=coldist(wordsleftT,wordsleftF,wordsleftS)
    return FL,SL



#Feature 2 word in position -1(last word in context)



def last(lastW,labels,i):
    for line in labels[i]:
        ind=len(line)-1
        temp=line[ind]
        if temp in lastW:
            lastW[temp] += 1
        else:
            lastW[temp]=1
    return lastW



def lastWord(labels,check):
    lastW={}
    lastWOT={}
    lastWB={}
    lastW=last(lastW,labels,0)
    lastW=last(lastW,labels,1)
    lastWOT=last(lastWOT,labels,0)
    lastWB=last(lastWB,labels,1)
    FL={}
    SL={}
    FL,SL=coldist(lastW,lastWOT,lastWB)
    return FL,SL
    

#Feature 3 POS in position +-1 left or right:

    
def posW(wordsleft,labels,i,k,o,check):
    for line in range(len(labels[i])):
        ind=labels[i][line].index(check)
        ind=ind-k
        x=o[line][ind][1]
        left=x
        if left in wordsleft:
            wordsleft[left] += 1
        else:
            wordsleft[left]=1
    return wordsleft

def posFeature(labels,k,check):
    
    o1=[]
    o2=[]
    for l in labels[0]:
        o1.append(nltk.pos_tag(l))
    for l in labels[1]:
        o2.append(nltk.pos_tag(l))
    pos={}
    posOT={}
    posB={}
    posOT=posW(posOT,labels,0,k,o1,check)
    posB=posW(posB,labels,1,k,o2,check)
    pos=posW(posOT,labels,0,k,o1,check)
    pos=posW(posB,labels,1,k,o2,check)
    FL={}
    SL={}
    FL,SL=coldist(pos,posOT,posB)
    return FL,SL


#Step 4: calculating log-likelihood

def logLike(otb,b):
    total=otb+b
    prN=otb/total
    prD=b/total
    likelihood=prN/prD
    logLike=log1p(likelihood)
    logLike=abs(logLike)
    return logLike

def log_helper(a,b,f1,stg):
    one=[]
    k1=[]
    #k2=[]
    two=[]

    for k in a:
        one.append(a[k])
        k1.append(k)
    for k in b:
        two.append(b[k])
        #k2.append(k)
    if k1[0] not in f1:
        f1[k1[0]+'O'+stg]=logLike(one[0],two[0])
    else:
        k1[0]=k1[0]+'x'
        f1[k1[0]+'O'+stg]=logLike(one[0],two[0])
    if k1[1] not in f1:
        f1[k1[1]+'B'+stg]=logLike(one[1],two[1])
    else:
        k1[1]=k1[1]+'x'
        f1[k1[1]+'B'+stg]=logLike(one[1],two[1])
    return f1

def decisionList(f1):
    d = OrderedDict(sorted(f1.items(), key=itemgetter(1)))
    l=list(d.items())[:]
    l=l[::-1]
    answer=[]
    rulepriority=[]
    rules=[]
    for line in l:
        x=line[0]
        rulepriority.append(x[len(x)-1:])
        x=x[:len(x)-1]
        answer.append(x[len(x)-1:])
        x=x[:len(x)-1]
        rules.append(x)
    return answer,rulepriority,rules



#Step 2: Training contexts
def train_context(new,check):
    final=[]
    for line in new:
        line=line.split()
        final.append(line)
    n=0
    labels=[[],[]]
    for line in final:
        if line[0] == check:
            n=n+1
            x=line[1:]
            labels[0].append(x)
        else:
            x=line[1:]
            labels[1].append(x)
    return labels

def train_helper(labels,check):
    leftOTB,leftB=wordK(labels,2,check)
    rightOTB,rightB=wordK(labels,-2,check)        
    lastOTB,lastB=lastWord(labels,check)    
    posleftOTB,posleftB=posFeature(labels,2,check)
    posrightOTB,posrightB=posFeature(labels,-2,check)        
    f1={}
    f1=log_helper(posleftOTB,posleftB,f1,'1')
    f1=log_helper(posrightOTB,posrightB,f1,'2')
    f1=log_helper(lastOTB,lastB,f1,'3')
    f1=log_helper(leftOTB,leftB,f1,'4')
    f1=log_helper(rightOTB,rightB,f1,'5')
    answer=[]
    rule_priority=[]
    rules=[]
    answer,rule_priority,rules=decisionList(f1)
    return answer,rule_priority,rules


def preprocess(file):
    l=[]
    global baseline
    f = open(file, "r")
    for line in f:
        line=line.strip('\n')
        line = re.sub('[!@#$,_.?:\'`?-]', '', line)    
        line = re.sub('\s+', ' ', line)
        l.append(line)
    t=0
    new=[]
    first=0
    second=0
    for line in l:
        if(line[0]=="*"):
            line=line.replace(line[0],"OT")
            t=t+1
            new.append(line)
            first=first+1
        else:
            second=second+1
            new.append(line)
    new = [w.lower() for w in new]
    total=first+second
    percentFirst=(first/total)*100
    percentSecond=(second/total)*100
    if percentFirst >= percentSecond:
        baseline='O'
    else:
        baseline='B'
    return new
    
def train(file,check1,check2):
    global ans
    global rulpr
    global rul
    new=[]
    new=preprocess(file)    
    labels=train_context(new,check2)
    answer=[]
    rule_priority=[]
    rules=[]
    answer,rule_priority,rules=train_helper(labels,check1)
    ans=answer
    rulpr=rule_priority
    rul=rules
    
    
def test_helper(result,now,baseline):
    lineresults=[]
    for line in now:
        pr= int(rulpr[0])
        if line[pr-1]==rul[0]:
            #result.append(baseline)
            result.append(ans[0])
        else:
            score=0
            d=int(rulpr.index('1'))
            if line[0]==rul[d]:
                if ans[d]=='O':
                    score=score+1
                else:
                    score=score-1
            d=int(rulpr.index('2'))
            if line[1]==rul[d]:
                if ans[d]=='O':
                    score=score+1
                else:
                    score=score-1
            d=int(rulpr.index('3'))
            if line[2]==rul[d]:
                if ans[d]=='O':
                    score=score+1
                else:
                    score=score-1
            d=int(rulpr.index('4'))
            if line[3]==rul[d]:
                if ans[d]=='O':
                    score=score+1
                else:
                    score=score-1
            d=int(rulpr.index('5'))
            if line[4]==rul[d]:
                if ans[d]=='O':
                    score=score+1
                else:
                    score=score-1
            if score > 0:
                #result.append(baseline)
                result.append('O')
            elif score < 0:
                #result.append(baseline)
                result.append('B')
            else:
                result.append(baseline)
    return result,lineresults

def test_part(testlabel,stg,i,baseline,check):
    now=[]
    c=0
    e=0
    for line in testlabel[i]:
        ind=line.index(check)
        left=line[ind-1]
        right=line[ind+1]
        last=line[len(line)-1]
        p=nltk.pos_tag(line)
        pl=p[ind-1][1]
        pr=p[ind+1][1]
        temp=[pl,pr,last,left,right]
        now.append(temp)
        result=[]    
    resultO,lineresults=test_helper(result,now,baseline)
    correct=[]
    wrong=[]
    for v in range(len(resultO)):
        if resultO[v] == stg:
            correct.append(testlabel[i][v])
            c=c+1
        else:
            e=e+1
            wrong.append(testlabel[i][v])
    return c
    
def test(file,check1,check2):
    testf=file
    testn=preprocess(testf)        
    testlabel=train_context(testn,check2)
    cO=test_part(testlabel,'O',0,baseline,check1)
    cB=test_part(testlabel,'B',1,baseline,check1)
    totalCorrect=cB+cO
    total=len(testlabel[0])+len(testlabel[1])
    accuracy=(totalCorrect/total)*100
    print("Accuracy: ",accuracy)
    

def main():
    import sys
    trainf=sys.argv[2]
    testf=sys.argv[4]
    trainf=trainf[1:len(trainf)-1]
    testf=testf[1:len(testf)-1]
    if trainf=="sake.trn":
        k='sake'
        m='otsake'
    elif trainf == "bass.trm":
        k='bass'
        m='otbass'
    train(trainf,k,m)
    test(testf,k,m)
    
main()
