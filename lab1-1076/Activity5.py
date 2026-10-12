
import pandas as pd
import matplotlib.pyplot as plt

# Load files
students = pd.read_csv("Machine-Learning-\lab1-1076\student.csv")
attendance = pd.read_json("Machine-Learning-\lab1-1076\attendance.json")
extra = pd.read_excel("Machine-Learning-\lab1-1076\extra.xlsx")

# Merge DataFrames
df = students.merge(attendance, on="Name")
df = df.merge(extra, on="Name")

print("Merged DataFrame:")
print(df)

# Identify students with attendance below 70%
low_attendance = df["Attendance"] < 70

# Scatter Plot
plt.figure(figsize=(8, 5))

# Students with attendance >= 70%
plt.scatter(
    df.loc[~low_attendance, "Marks"],
    df.loc[~low_attendance, "Attendance"],
    label="Attendance >= 70%"
)

# Highlight students with attendance < 70%
plt.scatter(
    df.loc[low_attendance, "Marks"],
    df.loc[low_attendance, "Attendance"],
    marker="x",
    s=100,
    label="Attendance < 70%"
)

plt.xlabel("Marks")
plt.ylabel("Attendance (%)")
plt.title("Marks vs Attendance")
plt.legend()
plt.grid(True)
plt.show()

# One-Hot Encoding on Name
encoded_df = pd.get_dummies(df, columns=["Name"])

print("\nOne-Hot Encoded DataFrame:")
print(encoded_df)