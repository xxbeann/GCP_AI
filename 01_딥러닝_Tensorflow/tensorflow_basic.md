# Linear Regression

**용어정리**   
* Linear Regression(선형 회귀) : 1차 함수, 직선의 방정식
* Weigh(가중치) : 입력변수가 출력에 미치는 영향, 기울기
* Bias(편향) : 기본 출력 값이 활성화 되는 정도, y절편
* Cost function(비용함수) : 2차 함수, 포물선의 방정식, (예측값 -실제값)^2, (H(x)-y)^2
* Hypothesis(예측 함수) : predict, H(x): 예측 값, y값: 답, x값: 입력 값(feature), H(x) = W * x + b
* 경사 하강법: 비용이 가장 적은 weigh를 구하는 알고리즘

### C언어 vs Python (객체지향 핵심)
* C, Java - 정통언어 // 메모리 기반의 프로그래밍, 변수가 바뀌면 값이 오버로드 되며 주소는 그대로 변하지 않음
* Python - 메모리 개념이 아니라 참조키의 역할, 값이 오버로드 되는게 아니라 참조키의 값만 바뀌는것<br>
파이썬은 모든 변수(함수)가 객체로 생성
numpy - 파이썬 성능이 안나오기때문에 c언어처럼 고속접근해서 사용할수 있게 끔 만들어줌(ex 데이터 처리)  
but, 단점 gpu를 못씀 -> gpu를 쓸 수 있게 만든게 구글의 tensor

### tf.Tensor: 텐서 객체
텐서는 다차원 배열로 넘파이 array와 비슷하며, tf.Tensor 객체는 데이터 타입과 크기르 가지고 있다.<br>
또한 tf.Tensor는 GPU같은 가속기 메모리에 상주할 수 있다.<br>
텐서플로는 텐서를 생성하고 이용하는 풍부한 연산 라이브러리(ft.add, tf.matmul, tf.linalg inv 등)를 제공하며<br>
연산수행 시 자동으로 텐서를 파이썬 네이티브 타입으로 변환하여 연산한다.

### Tensorflow
W(가중치) * X + b
AI - 규칙기반 -> 학습기반(머신러닝:ML)<br>머신러닝 알고리즘 중 하나가 딥러닝(인공신경망을 깊게 쌓아만듬)

### Tensorflow_Rank
1차원 shape : (3,)  
2차원 shape : (3, 3)  
3차원 shape : (2, 3, 3)  
4차원 shape : (1, 2, 3, 3)  
축 axis num = index num (ex 0, 1, 2, 3)  

### Tensorflow_shape
shape로 신경망을 만듬.
tenosr 객체가 주어진다는건 데이터가 주어진다는것 -> 데이터를 파악할 필요가 있음 특히 차원

### Multi Features
y = w * x + b  
multi variable feature  
y = w1 * x1 + w2 * x2 + b
cost function - 오차의 제곱을 데이터 개수로 나눔, 가설과 비용함수 사이의 오차를 구함  
cost(W,b) = (H(x) - y)^2 // H(x) = wx + b
b가 0이라면 W에 관한 2차함수 -> 미분계수가 0이될때 최소

**Ideas**<br>
- 택시 승하차 지점 분석 -> 시간대 사람 많이 모이는 곳 추천 시스템
