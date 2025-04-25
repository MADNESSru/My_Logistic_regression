from logistic_regression import LogisticRegression
from utils import load_data, accuracy
import matplotlib.pyplot as plt

# Загрузка данных
X_train, X_test, y_train, y_test = load_data()

# Инициализация и обучение модели
model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)

# Оценка точности
acc = accuracy(y_test, y_pred)
print(f'Точность модели: {acc:.4f}')

# Визуализация результатов (если 2 признака)
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', alpha=0.7)
plt.title('Логистическая регрессия (предсказания)')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.show()
