from src.data_preprocessing import load_and_prepare_data


data, X, y, X_train, X_test, y_train, y_test = load_and_prepare_data()

print("Dataset shape:", data.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nTraining set:", X_train.shape)
print("Test set:", X_test.shape)

print("\nTraining classes:")
print(y_train.value_counts())

print("\nTest classes:")
print(y_test.value_counts())