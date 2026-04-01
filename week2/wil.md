## Week2 WIL (What I Learned)

### 1. 목표

이번 주차에서는 도미와 빙어 데이터를 활용하여
KNN 모델을 학습하고, 데이터 전처리 및 평가 방법을 이해하는 것을 목표로 하였다.

특히,

* 데이터 분할(train/test split)
* 데이터 스케일링(표준화)
* 모델 성능 평가

과정을 직접 구현하는 데 중점을 두었다.

---

### 2. 데이터 분할 (Train/Test Split)

전체 데이터를 훈련 세트와 테스트 세트로 나누어 모델의 일반화 성능을 평가하였다.

* 7:3
* 5:5
* 9:1

세 가지 비율로 나누어 실험을 진행하였다.

```python
train_test_split(fish_data, fish_target, test_size=0.3, stratify=fish_target)
```

**핵심 개념**

* `test_size`로 비율 조절
* `stratify`를 사용하면 클래스 비율 유지 가능

**느낀 점**

* 데이터가 적을수록 분할 비율에 따라 성능 변동이 클 수 있음
* 특히 9:1은 테스트 데이터가 너무 적어 신뢰도가 떨어질 수 있음

---

### 3. 표준화 (Standardization)

KNN은 거리 기반 알고리즘이므로
특성(feature)의 스케일이 매우 중요하다.

이번 실습에서는 numpy 없이 직접 구현하였다.

#### 3.1 평균

[
mean = \frac{\sum x}{n}
]

#### 3.2 분산

[
variance = \frac{\sum (x - mean)^2}{n}
]

#### 3.3 표준편차

[
std = \sqrt{variance}
]

#### 3.4 표준화

[
z = \frac{x - mean}{std}
]

---

### 4. 스케일링 구현

```python
scaled_length = (fish[0] - mean_length) / std_length
scaled_weight = (fish[1] - mean_weight) / std_weight
```

**핵심**

* 평균 0, 표준편차 1로 변환
* 거리 계산 시 공정성 확보

**느낀 점**

* 단순한 공식이지만 모델 성능에 큰 영향을 줌
* 데이터 전처리가 모델보다 더 중요할 수 있음

---

### 5. 시각화 (Visualization)

스케일링된 데이터를 산점도로 시각화하여
도미와 빙어의 분포를 확인하였다.

```python
plt.scatter(bream_x, bream_y, marker='^', label='Bream')
plt.scatter(smelt_x, smelt_y, marker='o', label='Smelt')
```

**결과 해석**

* 도미는 길이와 무게가 큰 영역에 위치
* 빙어는 작은 영역에 밀집

**느낀 점**

* 시각화만으로도 두 클래스가 명확히 구분됨
* KNN이 높은 정확도를 가지는 이유를 직관적으로 이해 가능

---

### 6. KNN 모델 평가

각 분할 비율에서 KNN 정확도를 측정하였다.

```python
kn.fit(train_input, train_target)
kn.score(test_input, test_target)
```

**관찰**

* 대부분 높은 정확도 (1.0 근처)
* 데이터가 잘 분리되어 있기 때문

---

### 7. 전체 정리

이번 주차를 통해 다음을 이해하였다.

* 데이터 분할의 중요성
* 표준화의 필요성 (특히 거리 기반 모델)
* 시각화를 통한 데이터 이해
* KNN의 작동 원리

---

### 8. 느낀 점

* 단순한 모델(KNN)이라도 데이터가 잘 정리되면 높은 성능을 낼 수 있다.
* 모델보다 데이터 전처리가 더 중요하다는 것을 체감했다.
* 수식을 직접 구현하면서 개념 이해가 훨씬 깊어졌다.
* 앞으로는 단순히 라이브러리를 사용하는 것이 아니라 내부 동작을 이해하며 사용할 필요가 있다.
