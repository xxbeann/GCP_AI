# RNN

RNN(Recurrent Neural Network) : 순환 신경망   
순서가 있는 시퀀스 데이터, time series data(시계열 데이터)를 입력하여 예측


### 02_rnn_레이어종류

**RNN 주요 레이어 종류**
#### (1) SimpleRNN :가장 간단한 형태의 RNN레이어, 활성화 함수로 tanh가 사용됨(tanh: -1 ~ 1 사이의 값을 반환)
#### (2) LSTM(Long short Term Memory) : 입력 데이터와 출력 사이의 거리가 멀어질수로 연관 관계가 적어진다(Long Term Dependency,장기의존성 문제), LSTM은 장기 의존성 문제를 해결하기 위해 출력값외에 셀상태(cell state)값을 출력함, 활성화 함수로 tanh외에 sigmoid가 사용됨
#### (3) GRU(Gated Recurent Unit) : 뉴욕대 조경현 교수 등이 제안, LSTM보다 구조가 간단하고 성능이 우수함
