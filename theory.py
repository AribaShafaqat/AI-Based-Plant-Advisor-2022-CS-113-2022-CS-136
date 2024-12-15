import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score
# from sklearn.neural_network import MLPClassifier
# from scipy.sparse import hstack


# Read data from the CSV file
df = pd.read_csv('ok.csv', encoding='latin1')

# # Preprocess data: Convert text data to numeric using TF-IDF Vectorizer for 'Plant Name', 'Watering', 'Soil', and 'Fertilization Type'
# vectorizer_name = TfidfVectorizer(stop_words="english")
# X_name = vectorizer_name.fit_transform(df['Plant Name'])

# vectorizer_watering = TfidfVectorizer(stop_words="english")
# X_watering = vectorizer_watering.fit_transform(df['Watering'])

# vectorizer_soil = TfidfVectorizer(stop_words="english")
# X_soil = vectorizer_soil.fit_transform(df['Soil'])

# vectorizer_fertilization = TfidfVectorizer(stop_words="english")
# X_fertilization = vectorizer_fertilization.fit_transform(df['Fertilization Type'])

# # Combine all vectors into one feature matrix
# X = hstack([X_name, X_watering, X_soil, X_fertilization])

# # Encode target variables (Growth, Sunlight, Soil, Fertilization Type, and Watering)
# encoder_growth = LabelEncoder()
# y_growth = encoder_growth.fit_transform(df['Growth'])

# encoder_sunlight = LabelEncoder()
# y_sunlight = encoder_sunlight.fit_transform(df['Sunlight'])

# encoder_soil = LabelEncoder()
# y_soil = encoder_soil.fit_transform(df['Soil'])

# encoder_fertilization = LabelEncoder()
# y_fertilization = encoder_fertilization.fit_transform(df['Fertilization Type'])

# encoder_watering = LabelEncoder()
# y_watering = encoder_watering.fit_transform(df['Watering'])

# # Combine encoded targets into one target matrix
# y = pd.DataFrame({
#     'Growth': y_growth,
#     'Sunlight': y_sunlight,
#     'Soil': y_soil,
#     'Fertilization Type': y_fertilization,
#     'Watering': y_watering
# })

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Models to evaluate
# models = {
#     'Logistic Regression': MultiOutputClassifier(LogisticRegression()),
#     'Decision Tree': MultiOutputClassifier(DecisionTreeClassifier()),
#     'Random Forest': MultiOutputClassifier(RandomForestClassifier()),
#     'K-Nearest Neighbors': MultiOutputClassifier(KNeighborsClassifier()),
#     'Support Vector Machine': MultiOutputClassifier(SVC(probability=True)),
#     'Gradient Boosting': MultiOutputClassifier(GradientBoostingClassifier()),
#     'AdaBoost': MultiOutputClassifier(AdaBoostClassifier()),
#     'Naive Bayes': MultiOutputClassifier(MultinomialNB()),
#     'Neural Network': MultiOutputClassifier(MLPClassifier(max_iter=300)),
    
# }

# # Optional: Adding Stacking Classifier
# estimators = [
#     ('rf', RandomForestClassifier()),
#     ('gb', GradientBoostingClassifier()),
#     ('svc', SVC(probability=True))
# ]
# models['Stacking Classifier'] = MultiOutputClassifier(StackingClassifier(estimators=estimators, final_estimator=LogisticRegression()))

# # Evaluate each model
# for model_name, model in models.items():
#     print(f"Training {model_name}...")
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     # Calculate individual feature accuracies
#     growth_accuracy = accuracy_score(y_test['Growth'], y_pred[:, 0])
#     sunlight_accuracy = accuracy_score(y_test['Sunlight'], y_pred[:, 1])
#     soil_accuracy = accuracy_score(y_test['Soil'], y_pred[:, 2])
#     fertilization_accuracy = accuracy_score(y_test['Fertilization Type'], y_pred[:, 3])
#     watering_accuracy = accuracy_score(y_test['Watering'], y_pred[:, 4])

#     # Calculate total accuracy (average of all feature accuracies)
#     total_accuracy = (growth_accuracy + sunlight_accuracy + soil_accuracy + fertilization_accuracy + watering_accuracy) / 5

#     # Print results
#     print("Growth Accuracy:", growth_accuracy)
#     print("Sunlight Accuracy:", sunlight_accuracy)
#     print("Soil Accuracy:", soil_accuracy)
#     print("Fertilization Type Accuracy:", fertilization_accuracy)
#     print("Watering Accuracy:", watering_accuracy)
#     print("Total Accuracy:", total_accuracy)
#     print("---")



# # Check class distribution for each target column
# targets = {'Growth': y_growth, 'Sunlight': y_sunlight, 'Watering': y_watering, 'Soil': y_soil, 'Fertilization': y_fertilization}

# for name, target in targets.items():
#     print(f"Class distribution for '{name}':")
#     print(pd.Series(target).value_counts())
#     print()







import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import average_precision_score

# Load the dataset
df = pd.read_csv('ok.csv', encoding='latin1')

# Split the dataset into features (X) and target variables (y)
X = df['Plant Name']
y_growth = df['Growth']
y_sunlight = df['Sunlight']
y_watering = df['Watering']
y_soil = df['Soil']
y_fertilization = df['Fertilization Type']

# Vectorize the plant name (you can adjust this as necessary)
vectorizer = TfidfVectorizer(stop_words='english')

# Apply TF-IDF vectorizer to the 'Plant Name' feature
X_vectorized = vectorizer.fit_transform(X)

# Create separate models for Growth, Sunlight, Watering, Soil, and Fertilization
growth_model = RandomForestClassifier(class_weight='balanced')
sunlight_model = RandomForestClassifier(class_weight='balanced')
watering_model = RandomForestClassifier(class_weight='balanced')
soil_model = RandomForestClassifier(class_weight='balanced')
fertilization_model = RandomForestClassifier(class_weight='balanced')

# Handling Imbalance using RandomOversampling and TomekLinks
ros = RandomOverSampler(random_state=42)
tomek = TomekLinks()

# Split the data into train and test sets
X_train_watering, X_test_watering, y_watering_train, y_watering_test = train_test_split(X_vectorized, y_watering, test_size=0.3, random_state=42)
X_train_fertilization, X_test_fertilization, y_fertilization_train, y_fertilization_test = train_test_split(X_vectorized, y_fertilization, test_size=0.3, random_state=42)

# Apply Random Oversampling on the 'Watering' and 'Fertilization' targets
X_train_watering_sm, y_watering_train_sm = ros.fit_resample(X_train_watering, y_watering_train)
X_train_fertilization_sm, y_fertilization_train_sm = ros.fit_resample(X_train_fertilization, y_fertilization_train)

# Apply TomekLinks to remove the borderline examples for Watering and Fertilization
X_train_watering_tl, y_watering_train_tl = tomek.fit_resample(X_train_watering_sm, y_watering_train_sm)
X_train_fertilization_tl, y_fertilization_train_tl = tomek.fit_resample(X_train_fertilization_sm, y_fertilization_train_sm)

# Train the models for Growth, Sunlight, Soil (using original data)
growth_model.fit(X_vectorized, y_growth)
sunlight_model.fit(X_vectorized, y_sunlight)
soil_model.fit(X_vectorized, y_soil)

# Train the models for watering and fertilization after applying RandomOversampling and TomekLinks
watering_model.fit(X_train_watering_tl, y_watering_train_tl)
fertilization_model.fit(X_train_fertilization_tl, y_fertilization_train_tl)

# Step 1: Hyperparameter Tuning with GridSearchCV (for Watering and Fertilization models)
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
grid_search = GridSearchCV(watering_model, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_watering_tl, y_watering_train_tl)
watering_model = grid_search.best_estimator_

grid_search = GridSearchCV(fertilization_model, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_fertilization_tl, y_fertilization_train_tl)
fertilization_model = grid_search.best_estimator_

# Step 2: Cross-validation
scores_watering = cross_val_score(watering_model, X_train_watering_tl, y_watering_train_tl, cv=5)
scores_fertilization = cross_val_score(fertilization_model, X_train_fertilization_tl, y_fertilization_train_tl, cv=5)
print(f"Watering model cross-validation scores: {scores_watering}")
print(f"Fertilization model cross-validation scores: {scores_fertilization}")

# Step 3: Confusion Matrix and Classification Report
y_growth_pred = growth_model.predict(X_vectorized)
y_sunlight_pred = sunlight_model.predict(X_vectorized)
y_soil_pred = soil_model.predict(X_vectorized)

y_watering_pred = watering_model.predict(X_test_watering)
y_fertilization_pred = fertilization_model.predict(X_test_fertilization)

# Display Confusion Matrix and Classification Report for each model
print("Growth Model - Confusion Matrix and Classification Report:")
print(confusion_matrix(y_growth, y_growth_pred))
print(classification_report(y_growth, y_growth_pred))

print("Sunlight Model - Confusion Matrix and Classification Report:")
print(confusion_matrix(y_sunlight, y_sunlight_pred))
print(classification_report(y_sunlight, y_sunlight_pred))

print("Soil Model - Confusion Matrix and Classification Report:")
print(confusion_matrix(y_soil, y_soil_pred))
print(classification_report(y_soil, y_soil_pred))

print("Watering Model - Confusion Matrix and Classification Report:")
print(confusion_matrix(y_watering_test, y_watering_pred))
print(classification_report(y_watering_test, y_watering_pred))

print("Fertilization Model - Confusion Matrix and Classification Report:")
print(confusion_matrix(y_fertilization_test, y_fertilization_pred))
print(classification_report(y_fertilization_test, y_fertilization_pred))

# Step 4: Feature Importance (For Watering Model)
feature_importances = watering_model.feature_importances_
features = vectorizer.get_feature_names_out()
plt.barh(features, feature_importances)
plt.xlabel("Feature Importance")
plt.title("Feature Importance for Watering Prediction")
plt.show()

# Step 5: Test on Unseen Data (Using test set)
X_test = vectorizer.transform(df['Plant Name'])
y_test = df['Watering']
y_pred = watering_model.predict(X_test)
print(f"Accuracy on unseen test data: {accuracy_score(y_test, y_pred)}")

# Step 6: Precision-Recall Curve for Watering model
# Convert multiclass labels to binary format using One-vs-Rest strategy
lb = LabelBinarizer()
y_watering_bin = lb.fit_transform(y_watering_test)

# Get the predicted probabilities for each class
y_scores = watering_model.predict_proba(X_test_watering)

# Iterate over each class
for i in range(y_watering_bin.shape[1]):
    precision, recall, thresholds = precision_recall_curve(y_watering_bin[:, i], y_scores[:, i])
    average_precision = average_precision_score(y_watering_bin[:, i], y_scores[:, i])
    
    # Plot Precision-Recall curve for each class
    plt.plot(recall, precision, label=f'Class {i} (AP={average_precision:.2f})')

# Add labels and title to the plot
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Watering Model (Multiclass)')
plt.legend(loc='best')
plt.show()

# Step 7: Ensemble Models (Optional)
ensemble_model = VotingClassifier(estimators=[('rf_watering', watering_model), ('rf_fertilization', fertilization_model)], voting='hard')
ensemble_model.fit(X_train_watering_tl, y_watering_train_tl)
ensemble_y_pred = ensemble_model.predict(X_test_watering)
print(f"Ensemble Model Accuracy: {accuracy_score(y_watering_test, ensemble_y_pred)}")

# Step 8: Thresholding (Optional - Example to adjust decision threshold for Watering model)
threshold = 0.6  # Custom threshold
y_pred_threshold = (watering_model.predict_proba(X_test_watering)[:, 1] > threshold).astype(int)
print(f"Accuracy with custom threshold: {accuracy_score(y_watering_test, y_pred_threshold)}")

# Function to predict growth, sunlight, and watering for a given plant name
def predict_plant_info(plant_name):
    # Vectorize the input plant name
    plant_name_vec = vectorizer.transform([plant_name])
    
    # Predict for the input plant name
    growth_pred = growth_model.predict(plant_name_vec)[0]
    sunlight_pred = sunlight_model.predict(plant_name_vec)[0]
    watering_pred = watering_model.predict(plant_name_vec)[0]
    soil_pred = soil_model.predict(plant_name_vec)[0]
    fertilization_pred = fertilization_model.predict(plant_name_vec)[0]

    # Return predictions
    return growth_pred, sunlight_pred, watering_pred, soil_pred, fertilization_pred


# Example of how to use the function
plant_name = "Chrysanthemum"
growth, sunlight, watering, soil, fertilization = predict_plant_info(plant_name)
print(f"Predictions for {plant_name}: Growth: {growth}, Sunlight: {sunlight}, Watering: {watering}, Soil: {soil}, Fertilization: {fertilization}")



import joblib

# Save individual models
joblib.dump(growth_model, 'growth_model.pkl')
joblib.dump(sunlight_model, 'sunlight_model.pkl')
joblib.dump(soil_model, 'soil_model.pkl')
joblib.dump(watering_model, 'watering_model.pkl')
joblib.dump(fertilization_model, 'fertilization_model.pkl')

# Save the vectorizer (used for transforming plant names)
joblib.dump(vectorizer, 'vectorizer.pkl')
# Load the models
growth_model = joblib.load('growth_model.pkl')
sunlight_model = joblib.load('sunlight_model.pkl')
soil_model = joblib.load('soil_model.pkl')
watering_model = joblib.load('watering_model.pkl')
fertilization_model = joblib.load('fertilization_model.pkl')

# Load the vectorizer
vectorizer = joblib.load('vectorizer.pkl')
# Saving a model
joblib.dump(growth_model, 'growth_model.pkl')

# Loading the model
loaded_growth_model = joblib.load('growth_model.pkl')


print("Watering class distribution after oversampling:")
print(pd.Series(y_watering_train_sm).value_counts())

print("Fertilization class distribution after oversampling:")
print(pd.Series(y_fertilization_train_sm).value_counts())








# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import accuracy_score

# # Load the dataset
# df = pd.read_csv('ok.csv', encoding='latin1')


# # Split the dataset into features (X) and target variables (y)
# X = df['Plant Name']
# y_growth = df['Growth']
# y_sunlight = df['Sunlight']
# y_watering = df['Watering']
# y_soil=df['Soil']
# y_fertilization =df['Fertilization Type']

# # Vectorize the plant name (you can adjust this as necessary)
# vectorizer = TfidfVectorizer(stop_words='english')

# # Create separate models for Growth, Sunlight, and Watering
# growth_model = make_pipeline(vectorizer, RandomForestClassifier())
# sunlight_model = make_pipeline(vectorizer, RandomForestClassifier())
# watering_model = make_pipeline(vectorizer, RandomForestClassifier())
# soil_model = make_pipeline(vectorizer, RandomForestClassifier())
# fertilization_model = make_pipeline(vectorizer, RandomForestClassifier())


# # Train the models
# growth_model.fit(X, y_growth)
# sunlight_model.fit(X, y_sunlight)
# watering_model.fit(X, y_watering)
# soil_model.fit(X, y_soil)
# fertilization_model.fit(X, y_fertilization)


# # Function to predict growth, sunlight, and watering for a given plant name
# def predict_plant_info(plant_name):
#     # Predict for the input plant name
#     growth_pred = growth_model.predict([plant_name])[0]
#     sunlight_pred = sunlight_model.predict([plant_name])[0]
#     watering_pred = watering_model.predict([plant_name])[0]
#     soil_pred = soil_model.predict([plant_name])[0]
#     fertilization_pred = fertilization_model.predict([plant_name])[0]

#     # Return predictions
#     return growth_pred, sunlight_pred, watering_pred,soil_pred,fertilization_pred

# # Example of how to use the function
# plant_name = plant_name = input("Enter the name of the plant: ")
#   # You can replace this with any plant name
# growth, sunlight, watering,soil,fertilization = predict_plant_info(plant_name)

# print(f"Plant: {plant_name}")
# print(f"Growth: {growth}")
# print(f"Sunlight: {sunlight}")
# print(f"Watering: {watering}")
# print(f"Soil: {soil}")
# print(f"Fertilization: {fertilization}")