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

# 변수: tf.Variable()
# 초기 값이 미리 설정, 주로 텐서플로 내부에서 연산시 변경되는 변수에 사용 .weigh과 bias
a = tf.Variable(100)
b = tf.Variable(200)
c = tf.add(a,b)
print(c)
