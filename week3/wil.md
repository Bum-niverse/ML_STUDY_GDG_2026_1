# 🐟 Week 3 Assignment - 회귀 모델 및 규제

## 📌 1. 목표

이번 주차에서는 농어 데이터를 활용하여 분류가 아닌 **회귀 문제**를 해결하는 것을 목표로 하였다.

이전 주차에서는 KNN을 활용한 분류 문제를 다뤘다면,
이번에는 KNN을 회귀 문제에 적용해보고, 그 한계를 확인한 뒤
선형 회귀, 다항 회귀, 다중 회귀로 확장해보는 과정을 진행하였다.

또한, 모델의 복잡도에 따라 발생하는 과소적합과 과적합 문제를 이해하고,
이를 해결하기 위한 방법으로 **규제(Ridge, Lasso)**까지 적용해보았다.

즉, 단순히 모델을 사용하는 것이 아니라  
👉 모델을 개선하는 전체 흐름을 경험하는 것이 핵심 목표였다.

---

## 📌 2. 회귀 문제 정의

이번 실습에서는 농어의 길이를 기반으로 무게를 예측하는 회귀 문제를 다뤘다.

## 🔹 데이터 정의

```python
perch_length = np.array([...])
perch_weight = np.array([...])
```

## 🔹 개념 정리

* 회귀: 연속적인 값을 예측하는 문제
* 입력: 길이 (length)
* 출력: 무게 (weight)

---

## 📌 3. 데이터 시각화 및 분할

## 🔹 시각화 코드

```python
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

## 🔹 결과 해석

* 길이가 증가할수록 무게도 증가하는 경향 확인
* 대체로 선형적인 관계를 보임

---

## 🔹 데이터 분할

```python
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42
)
```

## 🔹 reshape

```python
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
```

## 🔹 개념 정리

* sklearn은 2차원 입력 필요
* (샘플 수, 특성 수) 형태로 변환해야 함

---

## 📌 4. KNN 회귀 모델

## 🔹 모델 학습

```python
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
knr.score(test_input, test_target)
```

## 🔹 MAE 계산

```python
from sklearn.metrics import mean_absolute_error

test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print(mae)
```

## 🔹 결과 해석

* 평균적으로 약 19g 정도 차이 발생
* 성능은 좋아 보이지만 한계 존재

---

## 🔹 KNN의 한계

```python
print(knr.predict([[50]]))
```

* 50cm 농어 → 약 1033g으로 예측

👉 실제보다 훨씬 작게 예측됨

## 🔹 이유

* 학습 데이터 최대 길이가 약 45cm
* 그 주변 데이터 평균을 사용하기 때문

👉 즉,
> KNN은 데이터 범위를 벗어난 예측이 불가능하다

---

## 📌 5. 과소적합 확인 및 개선

```python
print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))
```

* test > train → 과소적합

## 🔹 해결 방법

```python
knr.n_neighbors = 3
knr.fit(train_input, train_target)
```

## 🔹 개념 정리

* k ↓ → 모델 복잡도 증가
* 과소적합 완화

---

## 📌 6. 선형 회귀

## 🔹 모델 적용

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.predict([[50]]))
```

## 🔹 결과

* 50cm → 약 1241g

👉 KNN보다 더 현실적인 값

---

## 🔹 개념 정리

* 선형 회귀: `y = ax + b`
* coef: 기울기
* intercept: 절편
* 데이터 범위 밖 예측 가능

---

## 📌 7. 다항 회귀

## 🔹 특성 확장

```python
train_poly = np.column_stack((train_input**2, train_input))
test_poly = np.column_stack((test_input**2, test_input))
```

## 🔹 모델 학습

```python
lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]]))
```

## 🔹 결과

* 더 높은 값 예측 → 실제 데이터에 더 가까움

## 🔹 개념 정리

* 직선이 아닌 곡선 모델
* x² 같은 새로운 feature 추가

---

## 📌 8. 다중 회귀 및 특성공학

## 🔹 데이터 불러오기

```python
import pandas as pd

df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
```

## 🔹 PolynomialFeatures

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
```

## 🔹 개념 정리

* 여러 특성 사용 (길이, 높이, 두께)
* feature 자동 확장 (x², x₁x₂ 등)

---

## 📌 9. 과적합 발생

```python
poly = PolynomialFeatures(degree=5, include_bias=False)
```

## 🔹 결과

* train score ≈ 1
* test score 매우 낮음

👉 과적합 발생

## 🔹 개념 정리

* feature 많아질수록 모델 복잡도 증가
* 데이터 “암기” 상태

---

## 📌 10. 규제 (Regularization)

## 🔹 스케일링

```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
```

## 🔹 Ridge

```python
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_scaled, train_target)
```

## 🔹 Lasso

```python
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(train_scaled, train_target)
```

## 🔹 개념 정리

* Ridge: 계수를 작게 만듦
* Lasso: 일부 계수를 0으로 만듦

👉 모델 복잡도 조절

---

## 📌 11. 결과 및 비교

* KNN → 범위 밖 예측 불가
* Linear → 단순하지만 한계 존재
* Polynomial → 더 잘 맞지만 과적합 발생
* Ridge / Lasso → 과적합 해결

👉 모델은 하나로 해결되지 않음

---

## 📌 12. 느낀점

이번 실습을 통해 단순히 모델을 사용하는 것보다  
모델을 어떻게 개선하는지가 더 중요하다는 것을 느낄 수 있었다.

특히 KNN에서 시작해서 선형 회귀, 다항 회귀로 넘어가는 과정이
단순히 모델을 바꾸는 것이 아니라  
👉 “왜 이 모델이 필요한지”를 이해하는 과정이었다.

또한 feature를 늘리면 성능이 좋아질 것이라고 생각했지만,
오히려 과적합이 발생하는 것을 보면서  
모델의 복잡도를 조절하는 것이 중요하다는 것을 직접 확인할 수 있었다.

Ridge와 Lasso를 통해 과적합을 해결하는 과정을 보면서  
👉 머신러닝은 단순한 예측이 아니라 “조정의 과정”이라는 느낌을 받았다.

앞으로는 모델 하나에 집중하기보다는  
데이터, 모델, 전처리, 평가까지 전체 흐름을 같이 보는 습관을 가져야겠다고 느꼈다.
