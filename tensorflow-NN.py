import tensorflow as tf

# 神经网络前向传播

# 生命力两个变量，这里通过seed参数设定了随机种子，这样可以保证每次运行的结果都是一样的
a=tf.random_normal([2,3],stddev=1,seed=1)
sess=tf.Session()

print(sess.run(a))
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

# 暂时将输入的特征向量定义为常量，注意是1*2的矩阵
x=tf.constant([[0.7,0.9]])

# 前向传播求出神经网络的输出
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

with tf.Session() as sess:
    # 初始化w1和w2,如果不初始化就会报错
    sess.run(w1.initializer)
    sess.run(w2.initializer)
    print(sess.run(y))


print(tf.get_default_graph('y'))


import tensorflow as tf
x=tf.constant([[1,2],[2,3]])
# print(x)
y=tf.reduce_mean(x,0)
with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    print(sess.run(y))

import tensorflow as tf
import numpy as np
x = tf.constant([[1., 1.],
                 [2., 2.]])
tf.reduce_mean(x)  # 1.5
m1 = tf.reduce_mean(x,0)  # [1.5, 1.5]
m2 = tf.reduce_mean(x, 1)  # [1.,  2.]

xx = tf.constant([[[1., 1, 1],
                   [2., 2, 2]],

                  [[3, 3, 3],
                   [4, 4, 4]]])
m3 = tf.reduce_mean(xx, [0, 1]) # [2.5 2.5 2.5]


with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    print(sess.run(m1))
    print(sess.run(m2))

    print(xx.get_shape())
    print(sess.run(m3))

# 完整的神经网络tensorflow实现代码
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x=tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_=tf.placeholder(tf.float32,shape=(None,1),name='y-input')

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

y=tf.sigmoid(y)
# cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0))+(1-y)*tf.log(tf.clip_by_value((1-y,1e-10,1.0))))
# 定义损失熵和反向传播算法
cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdadeltaOptimizer(0.001).minimize(cross_entropy)
# 随机生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X=rdm.rand(dataset_size,2)
# 定义规则来给出样本标签，所有x1+x2<1的样本为正样本，而其他的为负样本，1为正样本，0为负样本
Y=[[int(x1+x2<1)] for (x1,x2) in X]

with tf.Session() as sess:
    # 最新版本的初始化方法
    # init_op=tf.global_variables_initializer()
    # 老版本的初始化，可能的原因是我安装的tensorflow的版本比较老
    init_op=tf.initialize_all_variables()
    sess.run(init_op)
    # tf.global_variables_initializer().run()
    # 在训练前神经网络参数的值
    print(sess.run(w1))
    print(sess.run(w2))
    # 设定训练次数
    STEEPS = 5000
    for i in range(STEEPS):
        # 每次选取batch_size个数据训练
        start = (i*batch_size)%dataset_size
        end = min(start+batch_size,dataset_size)
        # 通过选取的样本进行训练，并更新参数
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        # sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        # 每隔1000个数据进行记录
        if i%1000 ==0:
            # 每隔一段时间计算交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            # %(i,total_cross_entropy)之前不加逗号。
            print("after %d training step(s),cross entropy on all data is %g" %(i,total_cross_entropy))
            # print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
    # 通过训练之后得到的权值，该值跟一开始随机的取值不同
    print(sess.run(w1))
    print(sess.run(w2))
#
#
#
#
#
#
#
#
#
#


