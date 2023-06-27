# 03 cars_linear_regression_단항회귀
# X : 'speed' (속도), Y : 'dist' (제동거리)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.random.set_seed(5)

#데이터 읽어보기
xy = np.loadtxt('Tensorflow기본_데이터셋/data-01-cars.csv',delimiter=',',unpack=True)
# skiprows = 1 : 첫번째 행 건너 뛰기 
# unpack : x 따로, y따로
# delimiter : 변수를 어떻게 구분 할 것이냐 -> 쉼표 단위로 구분
# 넘파이는 같은 데이터셋만 읽음

x_train = xy[0]
y_train = xy[1]

# 변수 초기화 : weigh, bias
W = tf.Variable(tf.random.normal([1]),name='weigh')
b = tf.Variable(tf.random.normal([1]),name='bias')
print(W)
print(b)

# 예측 함수(hypothesis) : H(x) = W*X + b
def hypothesis(X):
    return X*W + b

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
for step in range(100001):
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
print('speed=30, dist=', hypothesis(30).numpy())
print('speed=50, dist=', hypothesis(50).numpy())
print('speed=[10,11,12,24,25], dist=', hypothesis([10,11,12,24,25]).numpy())

# 시각화 : matplotlib 사용
def prediction(X, W, b):
    return X * W + b
plt.plot(x_train,y_train,'ro') #red 원본데이터
plt.plot((0,25),(0,prediction(25,W,b)),'g') # green
plt.plot((0,25),(prediction(0,W,b),prediction(25,W,b)),'b') # blue 예측함수
