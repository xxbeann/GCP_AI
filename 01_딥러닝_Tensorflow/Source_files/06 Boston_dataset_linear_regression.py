import tensorflow as tf
import numpy as np
tf.random.set_seed(5)

#데이터 불러오기
# loadtxt로 불러오는 순간 np 배열로 묶어줌
train_xy = np.loadtxt('Tensorflow기본_데이터셋/boston_train.csv',delimiter=',',skiprows=1, dtype=np.float32)
test_xy = np.loadtxt('Tensorflow기본_데이터셋/boston_test.csv',delimiter=',',skiprows=1, dtype=np.float32)

# index slicing 
x_train = train_xy[:,:-1] # x, 마지막 컬럼을 제외
y_train = train_xy[:,[-1]] # y, 마지막 컬럼만 2차원으로 추출
x_test = test_xy[:,:-1]
y_test = test_xy[:,[-1]]

print(x_train.shape, y_train.shape)

# 변수 초기화 : weigh, bias

W = tf.Variable(tf.random.normal([9,1]),name='weigh')
b = tf.Variable(tf.random.normal([1]),name='bias')
print(W)
print(b)

# 예측 함수(hypothesis) : H(x) = W1*X1 + W2X2 + W3X3 + b

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

# weight 과 bias 출력
print('Weight:', W.numpy())
print("bias:", b.numpy())

# 예측
print('***** Predict')
print(hypothesis(x_test).numpy())

# 정확도 측정 : RMSE Root mean squared error
# 회귀모델의 평가
def get_rmse(y_test,preds):
    squared_error = 0
    for k,_ in enumerate(y_test):
        squared_error += (preds[k] - y_test[k])**2
    mse = squared_error/len(y_test)  
    rmse = np.sqrt(mse)
    return rmse[0]

preds = hypothesis(x_test).numpy()
print('RMSE:',get_rmse(y_test,preds))
