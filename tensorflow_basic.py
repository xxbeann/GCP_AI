import tensorflow as tf
tf.__version__

#상수 : tf.constant()
s = tf.constant("Hello Tensorflow")
print(s)
a = tf.constant([[1.,2.],
                 [3.,4.]])
b = tf.constant([[1.,1.],
                 [0.,1.]])
c = tf.matmul(a,b) # 내적 곱셈
print(c)
