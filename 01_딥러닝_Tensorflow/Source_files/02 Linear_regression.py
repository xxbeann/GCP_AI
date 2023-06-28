import tensorflow as tf
import numpy as np
tf.random.set_seed(5)

# 학습 데이터 : x와 y의 데이터
x_train = [1,2,3,4,5]
y_train = [1.1,2.2,3.3,4.4,5.5]

# 변수 초기화 : weigh, bias
W = tf.Variable(tf.random.normal([1]),name='weigh')
b = tf.Variable(tf.random.normal([1]),name='bias')
print(W)
print(b)

# 예측 함수(hypothesis) : H(x) = W*X + b
def hypothesis(X):
    return X*W + b
  
# 내장함수의 객체 속성 목록
# dir(hypothesis)

# 비용함수 : (H(x)-y)^2의 평균
# tf.square() : 제곱
# tf.reduce_mean() : 평균

def cost_func():
    cost = tf.reduce_mean(tf.square(hypothesis(x_train) - y_train))
    return cost

# 경사 하강법
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# 좀 더 발전된 경사 하강법
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# 학습 시작(W값을 계속 갱신)
print('**** Start learning!!')
for step in range(3000):
    optimizer.minimize(cost_func,var_list =[W,b]) # 비용함수, weigh, bias
    if step % 100 == 0:
        print('%04d'%step, 'cost:[',cost_func().numpy() ,']','W:',W.numpy(), 'b:', b.numpy())
print('**** Learning Finished!!')
# 이 셀만 실행시키면 학습횟수가 계속 올라감, 가중치와 바이어스를 초기화 시켜야 원하는 3천번 학습 가능

# weight 과 bias 출력
print('Weight:', W.numpy())
print("bias:", b.numpy())

# 예측
print('**** predict')
print('x=6.0, H(x)=',hypothesis(6.0).numpy())
print('x=9.5, H(x)=',hypothesis(9.5).numpy())
print('x=12.3, H(x)=',hypothesis(12.3).numpy())
