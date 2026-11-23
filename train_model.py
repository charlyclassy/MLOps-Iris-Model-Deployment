from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Test accuracy: {accuracy:.2%}")
model_path = Path(__file__).with_name('iris_model.joblib')
joblib.dump(model, model_path)
reloaded_model = joblib.load(model_path)
sample = X_test[0]
prediction = reloaded_model.predict([sample])[0]
print(f"Sample measurements: {sample.tolist()}")
print(f"Sample prediction: {iris.target_names[prediction]}")
