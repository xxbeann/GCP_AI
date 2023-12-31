# Supervised learning

### 용어정리 
* Linear Regression(선형 회귀) : 1차 함수, 직선의 방정식
* Weigh(가중치) : 입력변수가 출력에 미치는 영향, 기울기
* Bias(편향) : 기본 출력 값이 활성화 되는 정도, y절편
* Cost function(비용함수) : 2차 함수, 포물선의 방정식, (예측값 -실제값)^2, (H(x)-y)^2 의 평
* Hypothesis(예측 함수) : predict, H(x): 예측 값, y값: 답, x값: 입력 값(feature), H(x) = W * x + b
* 경사 하강법: 비용이 가장 적은 weigh를 구하는 알고리즘, 오차역전파 이용

### C언어 vs Python (객체지향 핵심)
* C, Java - 정통언어 // 메모리 기반의 프로그래밍, 변수가 바뀌면 값이 오버로드 되며 주소는 그대로 변하지 않음
* Python - 메모리 개념이 아니라 참조키의 역할, 값이 오버로드 되는게 아니라 참조키의 값만 바뀌는것<br>
파이썬은 모든 변수(함수)가 객체로 생성   
numpy - 파이썬 성능이 안나오기때문에 c언어처럼 고속접근해서 사용할수 있게 끔 만들어줌(ex 데이터 처리)  
but, 단점 gpu를 못씀 -> gpu를 쓸 수 있게 만든게 구글의 tensor

### tf.Tensor: 텐서 객체
텐서는 다차원 배열로 넘파이 array와 비슷하며, tf.Tensor 객체는 데이터 타입과 크기를 가지고 있다.<br>
또한 tf.Tensor는 GPU같은 가속기 메모리에 상주할 수 있다.<br>
텐서플로는 텐서를 생성하고 이용하는 풍부한 연산 라이브러리(ft.add, tf.matmul, tf.linalg inv 등)를 제공하며<br>
연산수행 시 자동으로 텐서를 파이썬 네이티브 타입으로 변환하여 연산한다.

### Tensorflow_Rank
1차원 shape : (3,)  
2차원 shape : (3, 3)  
3차원 shape : (2, 3, 3)  
4차원 shape : (1, 2, 3, 3)  
축 axis num = index num (ex 0, 1, 2, 3)  

### Tensorflow_shape
shape로 신경망을 만듬.
tenosr 객체가 주어진다는건 데이터가 주어진다는것 -> 데이터를 파악할 필요가 있음 특히 차원

### # Linear Regression
W(가중치) * X + b
AI - 규칙기반 -> 학습기반(머신러닝:ML)<br>머신러닝 알고리즘 중 하나가 딥러닝(인공신경망을 깊게 쌓아만듬)

### Multi Features
y = w * x + b  
multi variable feature  
y = w1 * x1 + w2 * x2 + b
cost function - 오차의 제곱을 데이터 개수로 나눔, 가설과 비용함수 사이의 오차를 구함  
cost(W,b) = (H(x) - y)^2 // H(x) = wx + b
b가 0이라면 W에 관한 2차함수 -> 미분계수가 0이될때 최소

### # Logistic Regression(binary classification)

답이 3개 이상이면 다중분류 2개면 이진분류

x -> wx + b -> sigmoid -> y <br>
Logistic Regression(binary classification)에 쓰이는 activation function은 sigmoid다.   
linear Regression에서는 activation function이 없다.   
<br>로지스틱 회귀는 선형 회귀 와는 다르게 종속 변수가 범주형 데이터를 대상으로 하며 입력 데이터가 주어졌을 때 해당 데이터의 결과가 특정 분류로 나뉘기 때문에 일종의 분류 (classification) 기법으로도 볼 수 있다.   
흔히 로지스틱 회귀는 종속변수가 이항형 문제(즉, 유효한 범주의 개수가 두개인 경우)를 지칭할 때 사용된다. 이외에, 두 개 이상의 범주를 가지는 문제가 대상인 경우엔 다항 로지스틱 회귀를 사용한다.

**actvation function 으로 sigmoid 사용**   
logloss - 로그함수를 사용한 이진분류모델   
C(H(x),y)=-ylog(H(X))-(1-y)long(1-H(x))

신경망이 1층일 때와 2층일때는 차원이 다르다. XOR문제 해결가능  

### # Multi Classification

softmax는 확률이기때문에 0과 1사이 값.
<br>multi-nomial classification (다중 분류) : Y값의 범주가 3개 이상인 분류   
**Activation function으로 softmax함수 사용**

### # ANN, DNN, CNN, RNN   
딥러닝은 인공신경망(Artificial Neural Network)ANN을 기초로 하고있다.   
ANN기법의 여러문제가 해결되면서 모델 내 은닉층을 많이 늘려서 학습의 결과를 향상시키는 방법이 등장하였고 이를 DNN(Deep Neural Network)라고 합니다. DNN은 은닉층을 2개이상 지닌 학습 방법을 뜻합니다.
https://ebbnflow.tistory.com/119   
기울기 손실 문제 때문에 은닉층에서는 relu함수 사용   
출력층은 sigmoid/softmax사용

### # Mnist Softmax

MNIST(Modified National Institute of Standard Technology) Dataset <br>
https://ko.wikipedia.org/wiki/MNIST<br>
label : 0 ~ 9 , 손글씨체 이미지  28*28(784 byte) , gray scale<br>
Train : 60000개 , Test : 10000개<br>

batch : 큰 데이터를 쪼개어 1회에 작은 단위로 가져다가 학습, next_batch()<br>
epoch : batch를 반복하여 전체 데이터가 모두 소진되었을 때를 1 epoch<br>
Vanishing Gradient  : 신경망이 깊어 질수록 입력신호가 사라진다(줄어든다), sigmoid 사용시<br>
Relu  : Rectified Linear Unit, DNN(deep neural net) 구현시 sigmoid 대신 사용됨<br>
dropout : 전체 신경망의 일부를 사용하지 않고 학습, 예측시는 전체를<br>

### # Result

softmax: 다중분류<br>
sigmoid: 2진분류<br>
no actioin function - 회귀<br>

**Ideas**<br>
- 택시 승하차 지점 분석 -> 시간대 사람 많이 모이는 곳 추천 시스템
