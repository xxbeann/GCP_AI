# CNN with Keras

### Convolutional Neural Net with Keras

**Keras Sequential layer의 주요 레이어**

1. Dense layer: FC layer, layer의 입력과 출력 사이에 있는 모든 뉴런이 서로 연결되는 layer
2. Flatten layer: 다차원을 단일(2차원)으로 축소하여 FC layer에 전달한다
3. Conv2D layer: 이미지 특징을 추출하는 Convolution Layer
4. MaxPool2D layer: 중요 데이터를 subsampling 하는 layer
5. Dropout layer: 학습시 신경망의 과적합을 막기위해 일부뉴런을 제거하는 layer

Reference: https://yeomko.tistory.com/40<br>
Xavier Glorot Initialization : W(Weight) 값을 fan_in,fan_out를 사용하여 초기화하여 정확도 향상<br>

**loss 종류**<br>
mean_squared_error : 평균제곱 오차<br>
binary_crossentropy : 이진분류 오차<br>
categorical_crossentropy : 다중 분류 오차. one-hot encoding 클래스, [0.2, 0.3, 0.5] 와 같은 출력값과 실측값의 오차값을 계산한다.<br>
sparse_categorical_crossentropy: 다중 분류 오차. 위와 동일하지만 , integer type 클래스라는 것이 다르다.<br>

**모델평가**<br>
Validation - 모델 내부의 파라미터를 검사<br>
Testing - 학습완료 후 검사<br>

**Image Augmentation**<br>
CNN은 영상의 2차원 변환인 회전, 크기, 밀림, 반사, 이동과 같은 2차원 변환인 Affine Transform에 취약하다.<br>
Affine Transform으로 변환된 영상은 다른 영상으로 인식한다.<br>
Data Augmentation을 위한 데이터 생성하는 방법으로는 keras의 ImageDataGenerator을 사용할 수 있다.

hiing
