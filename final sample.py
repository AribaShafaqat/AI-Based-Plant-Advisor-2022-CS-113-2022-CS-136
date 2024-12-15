# # import pandas as pd
# # from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# # from imblearn.under_sampling import RandomUnderSampler
# # from imblearn.over_sampling import SMOTE
# # import matplotlib.pyplot as plt

# # # Load the dataset
# # df = pd.read_csv('ok.csv', encoding='latin1')

# # # Split the dataset into features (X) and target variables (y)
# # X = df['Plant Name']
# # y_growth = df['Growth']
# # y_sunlight = df['Sunlight']
# # y_watering = df['Watering']
# # y_soil = df['Soil']
# # y_fertilization = df['Fertilization Type']

# # # Vectorize the plant name
# # vectorizer = TfidfVectorizer(stop_words='english')
# # X_vectorized = vectorizer.fit_transform(X)

# # # Create separate models for Growth, Sunlight, Watering, Soil, and Fertilization
# # growth_model = LogisticRegression(class_weight='balanced')
# # sunlight_model = RandomForestClassifier(class_weight='balanced')
# # watering_model = RandomForestClassifier(class_weight='balanced')
# # soil_model = RandomForestClassifier(class_weight='balanced')
# # fertilization_model = RandomForestClassifier(class_weight='balanced')

# # # Handling Imbalance for Watering using RandomUnderSampler
# # undersampler = RandomUnderSampler(random_state=42)
# # X_train_watering, X_test_watering, y_watering_train, y_watering_test = train_test_split(
# #     X_vectorized, y_watering, test_size=0.3, random_state=42
# # )
# # X_train_watering_us, y_watering_train_us = undersampler.fit_resample(X_train_watering, y_watering_train)

# # # Handling Imbalance for Fertilization using SMOTE
# # smote = SMOTE(random_state=42, k_neighbors=3)
# # X_train_fertilization, X_test_fertilization, y_fertilization_train, y_fertilization_test = train_test_split(
# #     X_vectorized, y_fertilization, test_size=0.3, random_state=42
# # )
# # X_train_fertilization_sm, y_fertilization_train_sm = smote.fit_resample(X_train_fertilization, y_fertilization_train)

# # # Train the models for Growth, Sunlight, and Soil (using original data)
# # growth_model.fit(X_vectorized, y_growth)
# # sunlight_model.fit(X_vectorized, y_sunlight)
# # soil_model.fit(X_vectorized, y_soil)

# # # Train the models for Watering and Fertilization after applying undersampling/oversampling
# # watering_model.fit(X_train_watering_us, y_watering_train_us)
# # fertilization_model.fit(X_train_fertilization_sm, y_fertilization_train_sm)

# # # Hyperparameter Tuning for Watering and Fertilization models
# # param_grid = {
# #     'n_estimators': [10, 50, 100],
# #     'max_depth': [5, 10, 15],
# #     'min_samples_split': [2, 5, 10],
# #     'min_samples_leaf': [1, 2, 4],
# #     'max_features': ['auto', 'sqrt', 'log2']
# # }

# # grid_search = GridSearchCV(watering_model, param_grid, cv=3, n_jobs=-1, verbose=2)
# # grid_search.fit(X_train_watering_us, y_watering_train_us)
# # watering_model = grid_search.best_estimator_

# # grid_search = GridSearchCV(fertilization_model, param_grid, cv=3, n_jobs=-1, verbose=2)
# # grid_search.fit(X_train_fertilization_sm, y_fertilization_train_sm)
# # fertilization_model = grid_search.best_estimator_

# # # Cross-validation
# # scores_watering = cross_val_score(watering_model, X_train_watering_us, y_watering_train_us, cv=5)
# # scores_fertilization = cross_val_score(fertilization_model, X_train_fertilization_sm, y_fertilization_train_sm, cv=5)
# # print(f"Watering model cross-validation scores: {scores_watering}")
# # print(f"Fertilization model cross-validation scores: {scores_fertilization}")

# # # Confusion Matrix and Classification Report
# # y_growth_pred = growth_model.predict(X_vectorized)
# # y_sunlight_pred = sunlight_model.predict(X_vectorized)
# # y_soil_pred = soil_model.predict(X_vectorized)

# # y_watering_pred = watering_model.predict(X_test_watering)
# # y_fertilization_pred = fertilization_model.predict(X_test_fertilization)

# # print("Growth Model - Confusion Matrix and Classification Report:")
# # print(confusion_matrix(y_growth, y_growth_pred))
# # print(classification_report(y_growth, y_growth_pred))

# # print("Sunlight Model - Confusion Matrix and Classification Report:")
# # print(confusion_matrix(y_sunlight, y_sunlight_pred))
# # print(classification_report(y_sunlight, y_sunlight_pred))

# # print("Soil Model - Confusion Matrix and Classification Report:")
# # print(confusion_matrix(y_soil, y_soil_pred))
# # print(classification_report(y_soil, y_soil_pred))

# # print("Watering Model - Confusion Matrix and Classification Report:")
# # print(confusion_matrix(y_watering_test, y_watering_pred))
# # print(classification_report(y_watering_test, y_watering_pred))

# # print("Fertilization Model - Confusion Matrix and Classification Report:")
# # print(confusion_matrix(y_fertilization_test, y_fertilization_pred))
# # print(classification_report(y_fertilization_test, y_fertilization_pred))

# # # Feature Importance (For Watering Model)
# # feature_importances = watering_model.feature_importances_
# # features = vectorizer.get_feature_names_out()
# # plt.barh(features, feature_importances)
# # plt.xlabel("Feature Importance")
# # plt.title("Feature Importance for Watering Prediction")
# # plt.show()

# # # Test on Unseen Data
# # X_test = vectorizer.transform(df['Plant Name'])
# # y_test = df['Watering']
# # y_pred = watering_model.predict(X_test)
# # print(f"Accuracy on unseen test data: {accuracy_score(y_test, y_pred)}")

# # # Function to predict growth, sunlight, watering, soil, and fertilization for a given plant name
# # def predict_plant_info(plant_name):
# #     plant_name_vec = vectorizer.transform([plant_name])
# #     growth_pred = growth_model.predict(plant_name_vec)[0]
# #     sunlight_pred = sunlight_model.predict(plant_name_vec)[0]
# #     watering_pred = watering_model.predict(plant_name_vec)[0]
# #     soil_pred = soil_model.predict(plant_name_vec)[0]
# #     fertilization_pred = fertilization_model.predict(plant_name_vec)[0]
# #     return growth_pred, sunlight_pred, watering_pred, soil_pred, fertilization_pred

# # # Example usage
# # plant_name = "Chrysanthemum"
# # growth, sunlight, watering, soil, fertilization = predict_plant_info(plant_name)
# # print(f"Plant: {plant_name}")
# # print(f"Growth: {growth}")
# # print(f"Sunlight: {sunlight}")
# # print(f"Watering: {watering}")
# # print(f"Soil: {soil}")
# # print(f"Fertilization: {fertilization}")
# # Necessary imports

# # from sklearn.model_selection import cross_val_score, StratifiedKFold
# # from sklearn.ensemble import RandomForestClassifier  # Example model
# # from sklearn.datasets import make_classification
# # from imblearn.over_sampling import RandomOverSampler
# # from collections import Counter

# # # Step 1: Generate synthetic dataset for demonstration (replace this with your actual dataset)
# # X, y = make_classification(
# #     n_samples=50,  # Adjust this to match your dataset size
# #     n_features=5,
# #     n_classes=3,
# #     n_clusters_per_class=1,
# #     weights=[0.1, 0.3, 0.6],  # Imbalanced class distribution
# #     random_state=42
# # )

# # # Step 2: Check class distribution
# # print("Original class distribution:", Counter(y))

# # # Step 3: Optional - Oversample the dataset to balance class distribution
# # oversampler = RandomOverSampler(random_state=42)
# # X_resampled, y_resampled = oversampler.fit_resample(X, y)
# # print("Class distribution after oversampling:", Counter(y_resampled))

# # # Step 4: Define the model
# # watering_model = RandomForestClassifier(random_state=42)

# # # Step 5: Use StratifiedKFold for balanced cross-validation
# # n_splits = 3  # Adjust this to be <= the minimum class size
# # skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# # # Step 6: Perform cross-validation
# # scores_watering = cross_val_score(watering_model, X_resampled, y_resampled, cv=skf)

# # # Step 7: Print results
# # print("Cross-validation scores:", scores_watering)
# # print("Mean cross-validation score:", scores_watering.mean())
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import SMOTE, RandomOverSampler
# from collections import Counter
# import matplotlib.pyplot as plt

# # Load the dataset
# df = pd.read_csv('ok.csv', encoding='latin1')

# # Split the dataset into features (X) and target variables (y)
# X = df['Plant Name']
# y_growth = df['Growth']
# y_sunlight = df['Sunlight']
# y_watering = df['Watering']
# y_soil = df['Soil']
# y_fertilization = df['Fertilization Type']

# # Vectorize the plant name
# vectorizer = TfidfVectorizer(stop_words='english')
# X_vectorized = vectorizer.fit_transform(X)

# # Create separate models for Growth, Sunlight, Watering, Soil, and Fertilization
# growth_model = LogisticRegression(class_weight='balanced')
# sunlight_model = RandomForestClassifier(class_weight='balanced')
# soil_model = RandomForestClassifier(class_weight='balanced')
# fertilization_model = RandomForestClassifier(random_state=42)

# # --------------------- Integrated Watering Model Handling ---------------------
# # Check the class distribution for "watering"
# print("Original class distribution for watering:", Counter(y_watering))

# # Oversample the dataset to balance the class distribution for "watering"
# oversampler = RandomOverSampler(random_state=42)
# X_resampled_watering, y_resampled_watering = oversampler.fit_resample(X_vectorized, y_watering)
# print("Class distribution after oversampling for watering:", Counter(y_resampled_watering))

# # Define the watering model
# watering_model = RandomForestClassifier(random_state=42)

# # Use StratifiedKFold for balanced cross-validation
# n_splits = 3  # Adjust this based on your dataset size
# skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# # Perform cross-validation for the watering model
# scores_watering = cross_val_score(watering_model, X_resampled_watering, y_resampled_watering, cv=skf)
# print("Cross-validation scores for watering model:", scores_watering)
# print("Mean cross-validation score for watering model:", scores_watering.mean())

# # Train the watering model on the resampled dataset
# watering_model.fit(X_resampled_watering, y_resampled_watering)

# # --------------------- Fertilization Model Handling ---------------------
# # Check the class distribution for "fertilization"
# print("Original class distribution for fertilization:", Counter(y_fertilization))

# # Oversample the dataset to balance the class distribution for "fertilization"
# X_resampled_fertilization, y_resampled_fertilization = oversampler.fit_resample(X_vectorized, y_fertilization)
# print("Class distribution after oversampling for fertilization:", Counter(y_resampled_fertilization))

# # Use StratifiedKFold for balanced cross-validation
# scores_fertilization = cross_val_score(fertilization_model, X_resampled_fertilization, y_resampled_fertilization, cv=skf)
# print("Cross-validation scores for fertilization model:", scores_fertilization)
# print("Mean cross-validation score for fertilization model:", scores_fertilization.mean())

# # Train the fertilization model on the resampled dataset
# fertilization_model.fit(X_resampled_fertilization, y_resampled_fertilization)

# # --------------------- Train Other Models ---------------------
# # Train the models for Growth, Sunlight, and Soil (using original data)
# growth_model.fit(X_vectorized, y_growth)
# sunlight_model.fit(X_vectorized, y_sunlight)
# soil_model.fit(X_vectorized, y_soil)

# # --------------------- Evaluation ---------------------
# # Confusion Matrix and Classification Report
# y_watering_pred = watering_model.predict(X_vectorized)
# y_fertilization_pred = fertilization_model.predict(X_vectorized)

# print("Watering Model - Confusion Matrix and Classification Report:")
# print(confusion_matrix(y_watering, y_watering_pred))
# print(classification_report(y_watering, y_watering_pred))

# print("Fertilization Model - Confusion Matrix and Classification Report:")
# print(confusion_matrix(y_fertilization, y_fertilization_pred))
# print(classification_report(y_fertilization, y_fertilization_pred))

# # --------------------- Feature Importance for Watering Model ---------------------
# feature_importances = watering_model.feature_importances_
# features = vectorizer.get_feature_names_out()
# plt.barh(features, feature_importances)
# plt.xlabel("Feature Importance")
# plt.title("Feature Importance for Watering Prediction")
# plt.show()

# # --------------------- Test on Unseen Data ---------------------
# def predict_plant_info(plant_name):
#     plant_name_vec = vectorizer.transform([plant_name])
#     growth_pred = growth_model.predict(plant_name_vec)[0]
#     sunlight_pred = sunlight_model.predict(plant_name_vec)[0]
#     watering_pred = watering_model.predict(plant_name_vec)[0]
#     soil_pred = soil_model.predict(plant_name_vec)[0]
#     fertilization_pred = fertilization_model.predict(plant_name_vec)[0]
#     return growth_pred, sunlight_pred, watering_pred, soil_pred, fertilization_pred

# # Example usage
# plant_name = "Chrysanthemum"
# growth, sunlight, watering, soil, fertilization = predict_plant_info(plant_name)
# print(f"Predictions for {plant_name} - Growth: {growth}, Sunlight: {sunlight}, Watering: {watering}, Soil: {soil}, Fertilization: {fertilization}")


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import shap  # For SHAP analysis
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('ok.csv', encoding='latin1')

# Split the dataset into features (X) and target variables (y)
X = df['Plant Name']
y_growth = df['Growth']
y_sunlight = df['Sunlight']
y_soil = df['Soil']

# Vectorize the plant name using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Optional: Scale the features if numerical features are added (not needed for text data but useful for other types of data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_vectorized.toarray())

# Train-Test Split for all target variables
X_train_growth, X_test_growth, y_train_growth, y_test_growth = train_test_split(X_scaled, y_growth, test_size=0.3, random_state=42)
X_train_sunlight, X_test_sunlight, y_train_sunlight, y_test_sunlight = train_test_split(X_scaled, y_sunlight, test_size=0.3, random_state=42)
X_train_soil, X_test_soil, y_train_soil, y_test_soil = train_test_split(X_scaled, y_soil, test_size=0.3, random_state=42)

# Apply SMOTE for balancing the training data (for all models)
smote = SMOTE(random_state=42)
X_train_growth, y_train_growth = smote.fit_resample(X_train_growth, y_train_growth)
X_train_sunlight, y_train_sunlight = smote.fit_resample(X_train_sunlight, y_train_sunlight)
X_train_soil, y_train_soil = smote.fit_resample(X_train_soil, y_train_soil)

# Create models with class weights
growth_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
sunlight_model = RandomForestClassifier(class_weight='balanced', random_state=42)
soil_model = RandomForestClassifier(class_weight='balanced', random_state=42)

# Hyperparameter tuning for RandomForest and LogisticRegression using GridSearchCV
param_grid_growth = {'C': [0.1, 1, 10], 'max_iter': [100, 1000]}
param_grid_sunlight = {'n_estimators': [100, 200, 300, 500], 'max_depth': [10, 20, 30, None], 'min_samples_split': [2, 5, 10]}
param_grid_soil = {'n_estimators': [100, 200, 300, 500], 'max_depth': [10, 20, 30, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}

grid_search_growth = GridSearchCV(estimator=growth_model, param_grid=param_grid_growth, cv=3)
grid_search_sunlight = GridSearchCV(estimator=sunlight_model, param_grid=param_grid_sunlight, cv=3)
grid_search_soil = GridSearchCV(estimator=soil_model, param_grid=param_grid_soil, cv=3)

# Train models with best parameters found via GridSearchCV
grid_search_growth.fit(X_train_growth, y_train_growth)
grid_search_sunlight.fit(X_train_sunlight, y_train_sunlight)
grid_search_soil.fit(X_train_soil, y_train_soil)

# Get the best models after hyperparameter tuning
growth_model_best = grid_search_growth.best_estimator_
sunlight_model_best = grid_search_sunlight.best_estimator_
soil_model_best = grid_search_soil.best_estimator_

# Train models on the full dataset with best hyperparameters
growth_model_best.fit(X_train_growth, y_train_growth)
sunlight_model_best.fit(X_train_sunlight, y_train_sunlight)
soil_model_best.fit(X_train_soil, y_train_soil)

# Predict on test data
y_pred_growth = growth_model_best.predict(X_test_growth)
y_pred_sunlight = sunlight_model_best.predict(X_test_sunlight)
y_pred_soil = soil_model_best.predict(X_test_soil)

# Evaluate models
print("Growth Model - Confusion Matrix and Classification Report:")
print(confusion_matrix(y_test_growth, y_pred_growth))
print(classification_report(y_test_growth, y_pred_growth))
print("Accuracy:", accuracy_score(y_test_growth, y_pred_growth))
print("F1-Score:", f1_score(y_test_growth, y_pred_growth, average='weighted'))
print("ROC-AUC:", roc_auc_score(y_test_growth, y_pred_growth, average='weighted', multi_class='ovr'))

print("\nSunlight Model - Confusion Matrix and Classification Report:")
print(confusion_matrix(y_test_sunlight, y_pred_sunlight))
print(classification_report(y_test_sunlight, y_pred_sunlight))
print("Accuracy:", accuracy_score(y_test_sunlight, y_pred_sunlight))
print("F1-Score:", f1_score(y_test_sunlight, y_pred_sunlight, average='weighted'))
print("ROC-AUC:", roc_auc_score(y_test_sunlight, y_pred_sunlight, average='weighted', multi_class='ovr'))

print("\nSoil Model - Confusion Matrix and Classification Report:")
print(confusion_matrix(y_test_soil, y_pred_soil))
print(classification_report(y_test_soil, y_pred_soil))
print("Accuracy:", accuracy_score(y_test_soil, y_pred_soil))
print("F1-Score:", f1_score(y_test_soil, y_pred_soil, average='weighted'))
print("ROC-AUC:", roc_auc_score(y_test_soil, y_pred_soil, average='weighted', multi_class='ovr'))

# Overall Model Accuracy (averaged across models)
overall_accuracy = (
    accuracy_score(y_test_growth, y_pred_growth) +
    accuracy_score(y_test_sunlight, y_pred_sunlight) +
    accuracy_score(y_test_soil, y_pred_soil)
) / 3

print("\nOverall Model Accuracy:", overall_accuracy)

# Combined Classification Report (for simplicity)
print("\nCombined Classification Report:")
print("Growth Model:\n", classification_report(y_test_growth, y_pred_growth))
print("Sunlight Model:\n", classification_report(y_test_sunlight, y_pred_sunlight))
print("Soil Model:\n", classification_report(y_test_soil, y_pred_soil))

# SHAP Analysis (optional but insightful)
# Explaining Soil Model predictions
explainer = shap.TreeExplainer(soil_model_best)
shap_values = explainer.shap_values(X_test_soil)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test_soil)
