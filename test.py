from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
print("Number of rows:", data.data.shape[0])
print("Number of columns:", data.data.shape[1])
print("Classes:", data.target_names)