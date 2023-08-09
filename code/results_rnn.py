from rnn_training_module import train_c2h4_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Assuming your data path is "../data/c2h4_final_data_phi_1.csv", call the function
model, _, _, X_test, y_test_res = train_c2h4_model("../data/c2h4_final_data_phi_1.csv")

# Get model predictions
predictions = np.argmax(model.predict(X_test), axis=1)

# Calculate and print metrics
accuracy = accuracy_score(np.argmax(y_test_res, axis=1), predictions)
precision = precision_score(np.argmax(y_test_res, axis=1), predictions, average='micro')
recall = recall_score(np.argmax(y_test_res, axis=1), predictions, average='micro')
f1 = f1_score(np.argmax(y_test_res, axis=1), predictions, average='micro')
classification_report_result = classification_report(np.argmax(y_test_res, axis=1), predictions)

print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Classification Report:\n", classification_report_result)

