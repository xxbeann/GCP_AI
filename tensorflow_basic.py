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

# 연산자
print(tf.add(1,2))
# 파이썬은 리스트 합 연산 불가 하지만 텐서는 가능
print(tf.add([1,2],[3,4]))
print(tf.square(5))
print(tf.reduce_sum([1,2,3]))

# 속성과 타입 변환
a = tf.constant([[1,2],
                [3,4]])
print(a)
print(type(a))
print(a.shape)
print(a.dtype)

# Tensor객체를 ndarray로 변환
n = a.numpy()
print(n, type(n)) # <class 'numpy.ndarray'>

# 파이썬 native 타입으로 변환
a = tf.constant(10)
b = int(a)
print(b, type(b))
