import pandas as pd

df = pd.read_csv("data/student_habits_performance.csv")
# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
print("Unique values")
print(df.nunique())
print("Categorical values")
categorical_columns = df.select_dtypes(include="str").columns
# for col in categorical_columns:
#     print(f"\n{col}")
#     print(df[col].value_counts())

# import matplotlib.pyplot as plt
# plt.hist(df['exam_score'],bins=20)
# plt.xlabel("Exam score")
# plt.ylabel("Number of students")
# plt.title("Exam score distribution")
# plt.show()
print(df['exam_score'].describe())
print(df['exam_score'].sort_values(ascending=False).head(20))
print((df['exam_score'] >= 95).sum())