import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "Monthly Salary": [15, 25, 18, 45, 35, 12, 38, 22, 30, 60],
    "Management Experience (years)": [0, 1, 0, 5, 3, 0, 4, 1, 2, 8],
    "Education (years)": [15, 16, 16, 18, 17, 14, 16, 15, 15, 19],
    "Weekly Work Hours": [40, 45, 40, 50, 45, 35, 45, 40, 42, 55],
    "Work Experience (years)": [2, 5, 3, 10, 7, 1, 8, 4, 6, 12]
}

df = pd.DataFrame(data)

X = df.drop("Monthly Salary", axis=1)
y = df["Monthly Salary"]

model = LinearRegression()
model.fit(X, y)

coefficients = pd.Series(model.coef_, index=X.columns)
intercept = model.intercept_

new_employee = pd.DataFrame({
    "Management Experience (years)": [2],
    "Education (years)": [16],
    "Weekly Work Hours": [43],
    "Work Experience (years)": [6]
})
predicted_salary = model.predict(new_employee)[0]

print("Model Coefficients:")
print(coefficients)
print(f"Intercept: {intercept:.2f}")
print(f"Predicted Salary for the New Employee: {predicted_salary:.2f} (thousands of ILS)")