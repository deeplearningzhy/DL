#coding=utf-8
from keras.models import Sequential
"""(1)创建一个模型实例"""
model=Sequential()#相当于创建一个模型实例

"""(2)利用一些网络层通过model.add()堆叠起来，就构建了一个网络模型"""
#从keras导入相关网络层，全连接层，激活层
from  keras.layers import Dense ,Activation
model.add(Dense(units=64,input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))

"""(3)完成模型的搭建后，我们要用.compile()方法来编译模型"""
model.compile(loss="categorical_crossentropy",
              optimizer='sgd',
              metrics=["accuracy"])#metrics:指标，标准，度量

"""(4)完成模型的编译后，我们在训练数据上按batch进行一定次数的迭代来训练网络"""
# model.fit(x_train,y_train,epochs=5,batch_size=32)#这里x_train,y_train应该是要提前定义啊，不然这里要飘红的

"""(5)随后，我们可以使用一行代码对我们的模型进行评估，看看模型的指标是否满足我们的要求"""
# loss_and_metrics=model.evaluate(x_text,y_test,batch_size=128)#看来这里的x_text,y_test也要提前定义的，不然也会飘红，那个
#这里的x_text,y_test应该是属于验证集把，不然应该不会有y_test的

"""(6)使用我们的模型，对新的数据进行预测"""
# classes=model.predict(x_test,batch_size=128)#这里的x_test要提前定义