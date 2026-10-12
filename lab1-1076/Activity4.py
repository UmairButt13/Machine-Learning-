import pandas as pd
import matplotlib.pyplot as plt

# Load files
students = pd.read_csv("Machine-Learning-\lab1-1076\student.csv")
attendance = pd.read_json("Machine-Learning-\lab1-1076\attendance.json")
extra = pd.read_excel("Machine-Learning-/lab1-1076/extra.xlsx")

# Merge DataFrames
df = students.merge(attendance, on="Name")
df = df.merge(extra, on="Name")

print("Merged DataFrame:")
print(df)

# Scatter Plot
low_attendance = df["Attendance"] < 70

plt.figure(figsize=(8, 5))

# Normal students
plt.scatter(
    df.loc[~low_attendance, "Marks"],
    df.loc[~low_attendance, "Attendance"],
    label="Attendance >= 50%"
)

# Highlight low attendance students
plt.scatter(
    df.loc[low_attendance, "Marks"],
    df.loc[low_attendance, "Attendance"],
    label="Attendance < 85%",
    marker="x",
    s=100
)

plt.xlabel("Marks")
plt.ylabel("Attendance (%)")
plt.title("Marks vs Attendance")
plt.legend()
plt.grid(True)
plt.show()

# One-Hot Encoding
encoded_df = pd.get_dummies(df, columns=["Name"])

print("\nOne-Hot Encoded DataFrame:")
print(encoded_df)