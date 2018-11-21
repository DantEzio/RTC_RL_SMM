# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 13:53:14 2018

@author: chong
"""

import numpy as np
import tensorflow as tf
import get_rpt
import set_datetime

import get_output#从out文件读取水动力初值
import set_pump#生成下一时段的inp
import change_rain#随机生成降雨序列

import random

from pyswmm import Simulation
    
'''
date_time=['08:00','08:30','09:00','09:30','10:00','10:30',\
           '11:00','11:30','12:00']
date_t=[0,30,60,90,120,150,180,210,240]

date_time=['08:00','08:20','08:40','09:00','09:20','09:40','10:00','10:20','10:40',\
           '11:00','11:20','11:40','12:00']
date_t=[0,20,40,60,80,100,120,140,160,180,200,220,240]

pump_list={'CC-storage':['CC-Pump-1','CC-Pump-2']}
limit_level=[0.9,3.02,0.9,3.02,0.9,1.26,1.53,1.71]
pool_name=['CC-storage','CC-storage','JK-storage','JK-storage','XR-storage','XR-storage','XR-storage','XR-storage']
max_depth=[5.6,4.8,7.72]
'''
date_time=['08:00','08:10','08:20','08:30','08:40','08:50',\
           '09:00','09:10','09:20','09:30','09:40','09:50',\
           '10:00','10:10','10:20','10:30','10:40','10:50',\
           '11:00','11:10','11:20','11:30','11:40','11:50','12:00']
date_t=[0,10,20,30,40,50,\
        60,70,80,90,100,110,\
        120,130,140,150,160,170,\
        180,190,200,210,220,230,240]


pump_list={'CC-storage':['CC-Pump-1','CC-Pump-2'],'JK-storage':['JK-Pump-1','JK-Pump-2'],'XR-storage':['XR-Pump-1','XR-Pump-2','XR-Pump-3','XR-Pump-4']}
limit_level={'CC-storage':[0.9,3.02,4.08],'JK-storage':[0.9,3.02,4.08],'XR-storage':[0.9,1.26,1.43,1.61,1.7]}
max_depth={'CC-storage':5.6,'JK-storage':4.8,'XR-storage':7.72}
pool_list=['CC-storage','JK-storage','XR-storage']


deltt=1

H=20
batch_size=3
learning_rate=1e-1
D=5
gamma=0.99

xs,ys,drs=[],[],[]
reward_sum=0
episode_number=0
total_episodes=12


def discount_reward(r):
    discounted_r=np.zeros_like(r)
    running_add=0
    for t in reversed(range(r.size)):
        running_add=running_add*gamma+r[t]
        discounted_r[t]=running_add
    return discounted_r

def simulation(filename):
    with Simulation(filename) as sim:
        #stand_reward=0
        for step in sim:
            pass    

def copy_result(outfile,infile):
    output = open(outfile, 'wt')
    with open(infile, 'rt') as data:
        for line in data:
            output.write(line)
    output.close()


#计算图
#For agent
tf.reset_default_graph()    
observations=tf.placeholder(tf.float32,[None,D],name="input_x")
W1=tf.get_variable("W1",shape=[D,H],initializer=tf.contrib.layers.xavier_initializer())
layer1=tf.nn.relu(tf.matmul(observations,W1))
W2=tf.get_variable("W2",shape=[H,1],initializer=tf.contrib.layers.xavier_initializer())

score=tf.matmul(layer1,W2)
probability=tf.nn.sigmoid(score)

input_y=tf.placeholder(tf.float32,[None,1],name="input_y")
advantages=tf.placeholder(tf.float32,name="reward_signal")
loglik=tf.log(input_y*(input_y-probability)+(1-input_y)*(input_y+probability))
loss=-tf.reduce_mean(loglik*advantages)

tvars=tf.trainable_variables()
newGrads=tf.gradients(loss,tvars)


adam=tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad=tf.placeholder(tf.float32,name="batch_grad1")
W2Grad=tf.placeholder(tf.float32,name="batch_grad2")
batchGrad=[W1Grad,W2Grad]
updateGrads=adam.apply_gradients(zip(batchGrad,tvars))


with tf.Session() as sess:
    rendering=False
    init=tf.global_variables_initializer()
    sess.run(init)    
    gradBuffer=sess.run(tvars)
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix]=grad*0
     
        
    old_reward=1000000
    while episode_number<=total_episodes:
        
        #每一次batch都新生成一个新的降雨
        A=random.randint(50,100)
        C=random.randint(3,9)/10.00
        #P=random.randint(1,5)
        #b=random.randint(1,3)
        #n=random.random()
        P=random.randint(1,5)
        b=12
        n=0.77
        R=random.randint(3,7)/10.00
        rain=change_rain.gen_rain(date_t[-1],A,C,P,b,n,R,deltt)
        
        
        infile='./sim/oti'
        startfile='./sim/ot'
        change_rain.copy_result(startfile+'.inp','arg-original.inp')
        change_rain.copy_result(infile+'.inp','arg-original.inp')
        
        
        change_rain.change_rain(rain,startfile+'.inp')
        print(A,C,P,b,n,R)
        
        sdate=edate='08/28/2015'
        #先sim10min
        stime=date_time[0]
        etime=date_time[1]
        set_datetime.set_date(sdate,edate,stime,etime,startfile+'.inp')
        
        change_rain.copy_result(infile+'.inp',startfile+'.inp')
        
        simulation(infile+'.inp')
        #获取rpt内信息，产生新的action
        total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(infile+'.rpt')
        #在确定泵开关之前确定最末时刻（当前）前池水位，水位过低时不开启
        pool_d=get_output.depth(infile+'.out',pool_list,date_t[1])
        rain_sum=sum(rain[date_t[0]:date_t[1]])/max(rain)
        
        action_seq=[]
        for i in range(1,len(date_t)-1):#用1场降雨生成结果,随机生成batch_size场降雨的inp

            action=[]
            pumps=[]
            for pool in pool_list:
                observation=[outflow/total_in,flooding/total_in,store/total_in,pool_d[pool],rain_sum]
                x_in=observation
                x=np.reshape(x_in,[1,D])
                tfprob=sess.run(probability,feed_dict={observations:x})    
                
                #初始化pump的action序列
                temaction=[]
                for pump in pump_list[pool]:
                    temaction.append(0)
                    
                flage=True
                if pool_d[pool]>(limit_level[pool][0]):
                    flage=True
                else:
                    flage=False

                if np.random.uniform()+0.5>tfprob and flage:
                    a=1
                    #设置pump
                    temaction[-1]=1
                    for j in range(len(limit_level[pool])-1):
                        if pool_d[pool]>=limit_level[pool][j+1]:
                            temaction[j]=1
                else:
                    a=0
                
                for item in temaction:
                    action.append(item)
                for item in pump_list[pool]:
                    pumps.append(item)
                xs.append(x)
                y=1-a
                #y=np.mean(a)
                ys.append(y)
                
            #设置pump并模拟之后才有reward
            action_seq.append(action)
            print(action_seq)
            #stime=date_time[i]
            etime=date_time[i+1]
            set_datetime.set_date(sdate,edate,stime,etime,startfile+'.inp')
            change_rain.copy_result(infile+'.inp',startfile+'.inp')
            set_pump.set_pump(action_seq,date_time[1:i+1],pumps,infile+'.inp')
            
            
            simulation(infile+'.inp')
            
            #获取rpt内信息，产生新的action
            total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(infile+'.rpt')
            #在确定泵开关之前确定最末时刻（当前）前池水位，水位过低时不开启
            pool_d=get_output.depth(infile+'.out',pool_list,date_t[i]-i)
            
            rain_sum=sum(rain[date_t[i]:date_t[i+1]])/max(rain)
            
            
            for pool in pool_list:
                reward=0
                if flooding/total_in>=0.5:
                    reward-=1.0
                else:
                    reward-=100
                
                if store/total_in>=0.0001:
                    reward-=1.0
                    
                if upflow<downflow:
                    reward+=10.0
                else:
                    reward-=100
                    
                
                if pool_d[pool]<=limit_level[pool][0] or pool_d[pool]>=max_depth[pool]:
                    reward-=10.0
            
                reward_sum+=reward
                drs.append(reward)
               
            
        episode_number+=1
            
        #完成一场降雨模拟，更新agent
        
        #when the game over, which means the stick fall; means the time is out in the new situation
        #记录一场降雨的reward
        epx=np.vstack(xs)
        epy=np.vstack(ys)
        epr=np.vstack(drs)
        xs,ys,drs=[],[],[]
        discounted_epr=discount_reward(epr)
        discounted_epr-=np.mean(discounted_epr)
        discounted_epr/=np.std(discounted_epr)

        tGrad=sess.run(newGrads,feed_dict={observations:epx,input_y:epy,advantages:discounted_epr})
        for ix,grad in enumerate(tGrad):
            gradBuffer[ix]+=grad
        
        #若已有一个batch的reward值，用于更新agent
        if episode_number%batch_size==0:
            print("train")
            sess.run(updateGrads,feed_dict={W1Grad:gradBuffer[0],W2Grad:gradBuffer[1]})
            for ix,grad in enumerate(gradBuffer):
                gradBuffer[ix]=grad*0
                print('Average reward for %d:%f.'%(episode_number,reward_sum/batch_size))
            #if reward_sum/batch_size>370:#reward标准待定
            if abs(old_reward-reward_sum/batch_size)/abs(old_reward)<=0.01:
                print("Task soveld in", episode_number)
                break
            old_reward=reward_sum/batch_size
            reward_sum=0
        #observation=env.reset()
        
    print("training done")
    
    
    
    #反复多次进行模拟
    for iten in range(2,8):
                
        infile='./sim/test/oti'
        startfile='./sim/test/ot'
        change_rain.copy_result(startfile+'.inp','arg-original.inp')
        change_rain.copy_result(infile+'.inp','arg-original.inp')
        
        #每一次batch都新生成一个新的降雨
        A=random.randint(50,80)
        C=random.randint(3,9)/10.00
        #P=random.randint(1,5)
        #b=random.randint(1,3)
        #n=random.random()
        P=random.randint(1,5)
        b=12
        n=0.77
        R=iten/10.00
        rain=change_rain.gen_rain(date_t[-1],A,C,P,b,n,R,deltt)
        
        change_rain.change_rain(rain,startfile+'.inp')
        print(A,C,P,b,n,R)
        
        
        #human
        simulation(startfile+'.inp')
        copy_result('./sim/test/result/en/inp/'+str(iten)+'.inp',startfile+'.inp')
        copy_result('./sim/test/result/en/rpt/'+str(iten)+'.rpt',startfile+'.rpt')
        
        
        #AI
        sdate=edate='08/28/2015'
        #先sim10min
        stime=date_time[0]
        etime=date_time[1]
        set_datetime.set_date(sdate,edate,stime,etime,startfile+'.inp')
        
        change_rain.copy_result(infile+'.inp',startfile+'.inp')
        
        simulation(infile+'.inp')
        #获取rpt内信息，产生新的action
        total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(infile+'.rpt')
        #在确定泵开关之前确定最末时刻（当前）前池水位，水位过低时不开启
        pool_d=get_output.depth(infile+'.out',pool_list,date_t[1])
        rain_sum=sum(rain[date_t[0]:date_t[1]])/max(rain)
        
        action_seq=[]
        for i in range(1,len(date_t)-1):#用1场降雨生成结果,随机生成batch_size场降雨的inp
 
            action=[]
            pumps=[]
            for pool in pool_list:
                observation=[outflow/total_in,flooding/total_in,store/total_in,pool_d[pool],rain_sum]
                x_in=observation
                x=np.reshape(x_in,[1,D])
                tfprob=sess.run(probability,feed_dict={observations:x})    
                
                #初始化pump的action序列
                temaction=[]
                for pump in pump_list[pool]:
                    temaction.append(0)
                    
                flage=True
                if pool_d[pool]>(limit_level[pool][0]):
                    flage=True
                else:
                    flage=False

                if np.random.uniform()+0.5>tfprob and flage:
                    a=1
                    #设置pump
                    temaction[-1]=1
                    for j in range(len(limit_level[pool])-1):
                        if pool_d[pool]>=limit_level[pool][j+1]:
                            temaction[j]=1
                else:
                    a=0
                
                for item in temaction:
                    action.append(item)
                for item in pump_list[pool]:
                    pumps.append(item)
                xs.append(x)
                y=1-a
                #y=np.mean(a)
                ys.append(y)
                
            #设置pump并模拟之后才有reward
            action_seq.append(action)
            
            #stime=date_time[i]
            etime=date_time[i+1]
            set_datetime.set_date(sdate,edate,stime,etime,startfile+'.inp')
            change_rain.copy_result(infile+'.inp',startfile+'.inp')
            set_pump.set_pump(action_seq,date_time[1:i+1],pumps,infile+'.inp')
            
            
            simulation(infile+'.inp')
            #change_rain.copy_result('check'+str(i)+'.inp',infile+'.inp')
            #获取rpt内信息，产生新的action
            total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(infile+'.rpt')
            #在确定泵开关之前确定最末时刻（当前）前池水位，水位过低时不开启
            pool_d=get_output.depth(infile+'.out',pool_list,date_t[i]-i)
            
            rain_sum=sum(rain[date_t[i]:date_t[i+1]])/max(rain)
            
            for pool in pool_list:
                reward=0
                if flooding/total_in>=0.5:
                    reward-=1.0
                else:
                    reward-=100
                
                if store/total_in>=0.0001:
                    reward-=1.0
                    
                if upflow<downflow:
                    reward+=10.0
                else:
                    reward-=100
                    
                
                if pool_d[pool]<=limit_level[pool][0] or pool_d[pool]>=max_depth[pool]:
                    reward-=10.0
            
                reward_sum+=reward

        #保存inp与rpt文件
        copy_result('./sim/test/result/ai/inp/'+str(iten)+'.inp','./sim/test/oti.inp')
        copy_result('./sim/test/result/ai/rpt/'+str(iten)+'.rpt','./sim/test/oti.rpt')
        print("操控序列：",action_seq)
        print("得分：",reward_sum)
         
    
