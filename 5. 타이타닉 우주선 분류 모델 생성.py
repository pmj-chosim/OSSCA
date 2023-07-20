import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# 데이터셋 로드
# 데이터셋은 Kaggle에서 다운로드하여 저장한 CSV 파일로 가정합니다. 데이터의 구조에 따라 아래 코드를 조정하세요.
data = pd.read_csv("titanic_data.csv")

# EDA 수행 (데이터 탐색 및 전처리)
# 데이터의 결측치 처리, 범주형 데이터 인코딩 등을 수행합니다. 필요한 전처리 과정을 추가하세요.

# 학습 데이터와 테스트 데이터 분할
X = data.drop("Survived", axis=1)
y = data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 분류 모델 생성 (Random Forest Classifier를 사용하겠습니다.)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("ROC AUC Score:", roc_auc)

# 모델을 파일로 저장
filename = 'titanic_model.pkl'
joblib.dump(model, filename)

# 저장한 모델 다시 로드
loaded_model = joblib.load(filename)

# 예측 결과 출력 (임의의 새로운 데이터로 예측)
# 새로운 데이터를 입력하고 예측 결과를 출력합니다. 필요에 따라 적절하게 수정하세요.
new_data = pd.DataFrame({
    "Feature1": [value1],
    "Feature2": [value2],
    # ...
})

prediction = loaded_model.predict(new_data)
print("Prediction:", prediction)
