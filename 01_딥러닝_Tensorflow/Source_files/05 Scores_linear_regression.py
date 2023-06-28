import tensorflow as tf
import numpy as np
tf.random.set_seed(5)

#데이터 불러오기
xy = np.loadtxt('Tensorflow기본_데이터셋/data-02-test-score.csv',delimiter=',',skiprows=1, dtype=np.float32)
# index slicing 
x_train = xy[:,:-1] # x, 마지막 컬럼을 제외
y_train = xy[:,[-1]] # y, 마지막 컬럼만 2차원으로 추출
print(x_train.shape, y_train.shape)

# 변수 초기화 : weigh, bias
# 다항회귀 일 때 변수 초기화와 예측 함수만 바뀜
# (m,n) * (n,L) = (m,L) : 행렬의 내적 곱셈
# (10,3) * (_,_) = (10,1) W -> (3,1)
# shape을 구할 수 있는지 여부

# (25,3) * (3,1) = (25,1)
# 바이어스는 뒤 1을 맞춰줘야함

W = tf.Variable(tf.random.normal([3,1]),name='weigh')
b = tf.Variable(tf.random.normal([1]),name='bias')
print(W)
print(b)

# 예측 함수(hypothesis) : H(x) = W1*X1 + W2X2 + W3X3 + b
# broadcasting : 갯수가 부족한 걸 맞춰줌 (repeat)
# (5,1) + (1,) = (5,1)
# 1로 쓰거나 없으면 broadcasting 다차원 배열 연산 시 작은쪽이 큰 쪽을 따라가게 해줌

def hypothesis(X):
    return tf.matmul(X,W) + b # 내적 곱셈

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
print('***** Predict')
x_data = [[73.,80.,75.],
          [93.,88.,93.],
          [89.,91.,90.],
          [96.,98.,100.],
          [73.,66.,70.]]
x_test = np.array(x_data,dtype=np.float32)
print(hypothesis(x_test).numpy())
# 원본
# 73,80,75,152
# 93,88,93,185
# 89,91,90,180
# 96,98,100,196
# 73,66,70,142

# 정확도 측정 : RMSE
def get_rmse(y_test,preds):
    squared_error = 0
    for k,_ in enumerate(y_test):
        squared_error += (preds[k] - y_test[k])**2
    mse = squared_error/len(y_test)  
    rmse = np.sqrt(mse)
    return rmse[0]

# 학습한 데이터를 그대로 검증 데이터로 사용한 경우
x_test = x_train
y_test = y_train

preds = hypothesis(x_test).numpy()
print('RMSE:',get_rmse(y_test,preds))  # RMSE: 2.4112918
