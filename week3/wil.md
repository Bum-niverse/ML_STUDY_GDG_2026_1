# 📌 Week 3 WIL (회귀 & 규제)

## 🧩 1. 이번 주 학습 흐름

이번 주는 2주차에서 배운 **지도학습(분류)**을 기반으로,  
👉 **회귀 문제와 모델 성능 개선 과정**을 학습했다.

전체 흐름:

KNN 회귀 → 한계 발견 → 선형 회귀 → 다항 회귀 → 과적합 발생 → 규제(Ridge, Lasso)

즉,

> 단순 모델 → 문제 발생 → 모델 개선 → 과적합 → 해결

이라는 머신러닝의 핵심 사이클을 경험하는 것이 목표였다.

---

## 🧠 2. 핵심 개념 정리

### ✔️ (1) 회귀 (Regression)
- 입력값을 통해 **연속적인 값**을 예측하는 문제
- 예: 길이 → 무게

---

### ✔️ (2) KNN 회귀
- 가까운 k개의 데이터 평균으로 예측

❗ 한계:
- 데이터 범위 밖 예측 불가 (extrapolation 불가)

---

### ✔️ (3) 선형 회귀
- 직선 모델: `y = ax + b`
- extrapolation 가능

---

### ✔️ (4) 다항 회귀
- 곡선 모델
- feature 확장 (x → x²)

---

### ✔️ (5) 과적합 / 과소적합

| 상태 | 특징 |
|------|------|
| 과소적합 | 모델이 너무 단순 |
| 과적합 | 모델이 데이터에 과하게 맞춤 |

---

### ✔️ (6) 규제 (Regularization)

| 방법 | 특징 |
|------|------|
| Ridge | 계수를 작게 유지 |
| Lasso | 일부 계수를 0으로 만듦 |

---

## 💻 3. 코드 리뷰 (셀별 흐름)

### 🔹 1. 데이터 정의

```python
perch_length = np.array([...])
perch_weight = np.array([...])
```

- 농어 길이와 무게 데이터를 정의  
- 회귀 문제의 input / target 역할  

---

### 🔹 2. 데이터 시각화

```python
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

- 길이와 무게의 관계 확인  
- 길이가 증가할수록 무게도 증가하는 경향 확인  

---

### 🔹 3. train/test 분리

```python
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42
)
```

- 데이터를 훈련/테스트로 분리  
- 모델의 일반화 성능 평가 목적  

---

### 🔹 4. reshape

```python
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
```

- sklearn은 2차원 입력을 요구  
- (샘플 수, 특성 수) 형태로 변환  

---

### 🔹 5. KNN 회귀 모델

```python
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
knr.score(test_input, test_target)
```

- KNN 기반 회귀 모델 학습  
- R² score로 성능 평가  

---

### 🔹 6. MAE 계산

```python
from sklearn.metrics import mean_absolute_error

test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print(mae)
```

- 평균 절댓값 오차 계산  
- 실제 값과 평균적으로 얼마나 차이 나는지 확인  

---

### 🔹 7. 과소적합 확인

```python
print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))
```

- test 점수가 train보다 높음 → 과소적합  
- 모델이 충분히 학습되지 못한 상태  

---

### 🔹 8. k 값 조정

```python
knr.n_neighbors = 3
knr.fit(train_input, train_target)
```

- k 값을 줄여 모델 복잡도 증가  
- 과소적합 완화  

---

### 🔹 9. 선형 회귀

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.predict([[50]]))
```

- 직선 기반 모델 학습  
- 데이터 범위 밖 값도 예측 가능  

---

### 🔹 10. 다항 회귀

```python
train_poly = np.column_stack((train_input**2, train_input))
test_poly = np.column_stack((test_input**2, test_input))
```

- x² 특성 추가  
- 직선이 아닌 곡선 형태 모델 생성  

---

### 🔹 11. 다중 회귀 (데이터 확장)

```python
import pandas as pd

df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
```

- 길이, 높이, 두께 등 여러 특성 사용  

---

### 🔹 12. PolynomialFeatures

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
```

- 자동으로 다항 특성 생성  
- x², x₁x₂ 등 추가  

---

### 🔹 13. 과적합 발생

```python
poly = PolynomialFeatures(degree=5, include_bias=False)
```

- 특성 수 급증  
- train 성능은 높지만 test 성능 급락 → 과적합  

---

### 🔹 14. 스케일링

```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
```

- 데이터 스케일 정규화  
- 큰 값의 영향 감소  

---

### 🔹 15. Ridge / Lasso

```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge()
ridge.fit(train_scaled, train_target)

lasso = Lasso()
lasso.fit(train_scaled, train_target)
```

- 규제를 통해 과적합 방지  
- 모델의 일반화 성능 향상  

---

## 🔍 4. 2주차와의 연결

| 항목 | 2주차 | 3주차 |
|------|------|------|
| 문제 | 분류 | 회귀 |
| 모델 | KNN (분류) | KNN, Linear, Polynomial |
| 목표 | 예측 | 성능 개선 |
| 평가 | Accuracy | R², MAE |

👉 핵심 변화:

> “모델 사용” → “모델 이해 및 조정”

---

## 💡 5. 배운 점

### ✔️ 1. 모델마다 한계가 존재한다
- KNN → extrapolation 불가  
- Linear → 직선 한계  
- Polynomial → 과적합 발생  

---

### ✔️ 2. 성능 평가는 반드시 필요하다
- train/test 비교 중요  
- R² + MAE 함께 확인 필요  

---

### ✔️ 3. 모델 복잡도 조절이 핵심이다
- 단순 → 과소적합  
- 복잡 → 과적합  

👉 적절한 균형 필요  

---

### ✔️ 4. Feature Engineering의 중요성
- 모델보다 입력 데이터가 더 중요할 수 있음  

---

### ✔️ 5. 규제는 필수 개념
- 실제 ML 모델에서 거의 항상 사용됨  

---

## 🧠 6. 한 줄 정리

> 이번 주는 “모델을 만드는 것”이 아니라  
> **“모델을 어떻게 개선하는지”를 배우는 과정이었다.**
