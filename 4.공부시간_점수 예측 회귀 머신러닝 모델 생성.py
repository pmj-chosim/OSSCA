import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# CSV 데이터 로드
data = pd.DataFrame({'hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     'score': [60, 63, 64, 67, 68, 71, 72, 75, 76, 78]})

# 데이터 시각화
plt.scatter(data['hours'], data['score'])
plt.xlabel('Study Hours')
plt.ylabel('Score')
plt.title('Study Hours vs. Score')
plt.show()

# 데이터 분할
X = data[['hours']]
y = data['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 생성
model = LinearRegression()
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

# 모델을 파일로 저장
filename = 'study_score_model.pkl'
joblib.dump(model, filename)

# 저장한 모델 다시 로드
loaded_model = joblib.load(filename)

# 예측 결과 출력
hours_to_predict = [12, 14]
for hours in hours_to_predict:
    predicted_score = loaded_model.predict([[hours]])
    print(f"{hours} hours of study is predicted to score: {predicted_score[0]:.2f}")
