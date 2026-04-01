# 🐟 Week 2 Assignment - 데이터 분할 및 수동 표준화

## 📌 1. 목표

이번 주차에서는 도미와 빙어 데이터를 활용하여 훈련 세트와 테스트 세트를 직접 분리하고,
데이터 전처리 과정 중 하나인 표준화(Standardization)를 numpy 없이 직접 구현해보는 것을 목표로 하였다.

또한, 분할 비율에 따른 KNN 모델의 성능 변화를 확인하고,
시각화를 통해 데이터 분포를 직관적으로 이해하고자 하였다.

---

## 📌 2. 데이터 분할 (Train/Test Split)

전체 데이터를 훈련 세트와 테스트 세트로 나누어 모델의 성능을 평가하였다.

이번 실습에서는 아래와 같은 세 가지 비율로 데이터를 분리하였다.

* 7:3
* 5:5
* 9:1

## 🔹 분할 코드

```python
train_input_7, test_input_7, train_target_7, test_target_7 = train_test_split(
    fish_data, fish_target, test_size=0.3, stratify=fish_target
)
```

## 🔹 개념 정리

* `test_size` : 테스트 데이터 비율 설정
* `stratify` : 클래스 비율 유지 (도미/빙어 비율 유지)

## 🔹 결과

* 7:3 → 훈련 34개 / 테스트 15개
* 5:5 → 훈련 24개 / 테스트 25개
* 9:1 → 훈련 44개 / 테스트 5개

---

## 📌 3. 수동 표준화 구현

이전 주차에서는 `StandardScaler`를 사용했지만,
이번에는 numpy 없이 직접 평균과 표준편차를 계산하여 표준화를 수행하였다.

## 🔹 평균 계산

```python
mean_length = sum(lengths) / n
mean_weight = sum(weights) / n
```

## 🔹 분산 계산

```python
var_length = sum((x - mean_length) ** 2 for x in lengths) / n
var_weight = sum((x - mean_weight) ** 2 for x in weights) / n
```

## 🔹 표준편차

```python
std_length = var_length ** 0.5
std_weight = var_weight ** 0.5
```

## 🔹 표준화

```python
scaled_length = (fish[0] - mean_length) / std_length
scaled_weight = (fish[1] - mean_weight) / std_weight
```

## 🔹 개념 정리

* 표준화 = (값 - 평균) / 표준편차
* 평균 0, 표준편차 1로 변환
* 거리 기반 모델(KNN)에서 매우 중요

---

## 📌 4. 시각화 및 데이터 분포 확인

표준화된 데이터를 산점도로 시각화하여 도미와 빙어의 분포를 확인하였다.

## 🔹 시각화 코드

```python
plt.scatter(bream_x, bream_y, color='red', marker='^', label='Bream (도미)')
plt.scatter(smelt_x, smelt_y, color='blue', marker='o', label='Smelt (빙어)')
plt.legend()
```

## 🔹 결과 해석

* 도미 → 오른쪽 위 (길고 무거움)
* 빙어 → 왼쪽 아래 (짧고 가벼움)

두 클래스가 명확하게 분리되는 것을 확인할 수 있었다.

---

## 📌 5. 결과 및 비교

각 분할 비율에 따라 KNN 모델의 정확도를 확인하였다.

* 대부분 높은 정확도 (1.0 근처)
* 데이터가 명확히 구분되기 때문

하지만 분할 비율에 따라 테스트 데이터의 개수가 달라지므로
모델 평가의 신뢰도가 달라질 수 있다.

특히 9:1의 경우 테스트 데이터가 너무 적어 평가가 불안정할 수 있다.

---

## 📌 6. 느낀점

이번 실습을 통해 데이터 전처리의 중요성을 다시 한 번 느낄 수 있었다.

특히, 이전에는 단순히 `StandardScaler`를 사용했지만,
이번에는 직접 평균과 표준편차를 계산하면서 표준화의 원리를 더 깊게 이해할 수 있었다.

또한, 데이터 분할 비율에 따라 모델 성능의 신뢰도가 달라질 수 있다는 점도 알게 되었다.

시각화를 통해 데이터 분포를 직접 확인하면서,
KNN이 왜 높은 정확도를 가지는지도 직관적으로 이해할 수 있었다.

앞으로는 단순히 모델을 사용하는 것이 아니라,
데이터 전처리와 평가 방법까지 함께 고려해야겠다고 느꼈다.

