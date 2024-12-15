import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


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
X_train_growth, X_test_growth, y_growth_train, y_growth_test = train_test_split(X_vectorized, y_growth, test_size=0.3, random_state=42)
X_train_sunlight, X_test_sunlight, y_sunlight_train, y_sunlight_test = train_test_split(X_vectorized, y_sunlight, test_size=0.3, random_state=42)
X_train_soil, X_test_soil, y_soil_train, y_soil_test = train_test_split(X_vectorized, y_soil, test_size=0.3, random_state=42)

# Apply Random Oversampling on the 'Watering', 'Fertilization', 'Growth', 'Sunlight', and 'Soil' targets
X_train_watering_sm, y_watering_train_sm = ros.fit_resample(X_train_watering, y_watering_train)
X_train_fertilization_sm, y_fertilization_train_sm = ros.fit_resample(X_train_fertilization, y_fertilization_train)
X_train_growth_sm, y_growth_train_sm = ros.fit_resample(X_train_growth, y_growth_train)
X_train_sunlight_sm, y_sunlight_train_sm = ros.fit_resample(X_train_sunlight, y_sunlight_train)
X_train_soil_sm, y_soil_train_sm = ros.fit_resample(X_train_soil, y_soil_train)

# Apply TomekLinks to remove the borderline examples for Watering, Fertilization, Growth, Sunlight, and Soil
X_train_watering_tl, y_watering_train_tl = tomek.fit_resample(X_train_watering_sm, y_watering_train_sm)
X_train_fertilization_tl, y_fertilization_train_tl = tomek.fit_resample(X_train_fertilization_sm, y_fertilization_train_sm)
X_train_growth_tl, y_growth_train_tl = tomek.fit_resample(X_train_growth_sm, y_growth_train_sm)
X_train_sunlight_tl, y_sunlight_train_tl = tomek.fit_resample(X_train_sunlight_sm, y_sunlight_train_sm)
X_train_soil_tl, y_soil_train_tl = tomek.fit_resample(X_train_soil_sm, y_soil_train_sm)

# Train the models for Growth, Sunlight, Soil (using original data)
growth_model.fit(X_train_growth_tl, y_growth_train_tl)
sunlight_model.fit(X_train_sunlight_tl, y_sunlight_train_tl)
soil_model.fit(X_train_soil_tl, y_soil_train_tl)

# Train the models for watering and fertilization after applying RandomOversampling and TomekLinks
watering_model.fit(X_train_watering_tl, y_watering_train_tl)
fertilization_model.fit(X_train_fertilization_tl, y_fertilization_train_tl)

# Step 1: Hyperparameter Tuning with GridSearchCV (for all models)
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

def tune_model(model, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Apply GridSearchCV to each model
growth_model = tune_model(growth_model, X_train_growth_tl, y_growth_train_tl)
sunlight_model = tune_model(sunlight_model, X_train_sunlight_tl, y_sunlight_train_tl)
soil_model = tune_model(soil_model, X_train_soil_tl, y_soil_train_tl)
watering_model = tune_model(watering_model, X_train_watering_tl, y_watering_train_tl)
fertilization_model = tune_model(fertilization_model, X_train_fertilization_tl, y_fertilization_train_tl)

# Step 2: Cross-validation for all models
models = [growth_model, sunlight_model, soil_model, watering_model, fertilization_model]
model_names = ['Growth', 'Sunlight', 'Soil', 'Watering', 'Fertilization']

for model, name in zip(models, model_names):
    scores = cross_val_score(model, X_train_watering_tl, y_watering_train_tl, cv=5)
    print(f"{name} model cross-validation scores: {scores}")

# Step 3: Generate Classification Reports for all models
y_growth_pred = growth_model.predict(X_test_growth)
y_sunlight_pred = sunlight_model.predict(X_test_sunlight)
y_soil_pred = soil_model.predict(X_test_soil)
y_watering_pred = watering_model.predict(X_test_watering)
y_fertilization_pred = fertilization_model.predict(X_test_fertilization)

# Generate classification reports for all models
growth_report = classification_report(y_growth_test, y_growth_pred)
sunlight_report = classification_report(y_sunlight_test, y_sunlight_pred)
soil_report = classification_report(y_soil_test, y_soil_pred)
watering_report = classification_report(y_watering_test, y_watering_pred)
fertilization_report = classification_report(y_fertilization_test, y_fertilization_pred)

# Combine all reports into a single report
overall_report = f"""
Growth Model Classification Report:
{growth_report}

Sunlight Model Classification Report:
{sunlight_report}

Soil Model Classification Report:
{soil_report}

Watering Model Classification Report:
{watering_report}

Fertilization Model Classification Report:
{fertilization_report}
"""

# Save the overall report to a text file
with open('overall_classification_report.txt', 'w') as file:
    file.write(overall_report)

# Print the overall report to console
print(overall_report)

# Step 4: Feature Importance (For all models)
def plot_feature_importance(model, model_name):
    feature_importances = model.feature_importances_
    features = vectorizer.get_feature_names_out()
    plt.barh(features, feature_importances)
    plt.xlabel("Feature Importance")
    plt.title(f"Feature Importance for {model_name} Prediction")
    plt.show()

# Plot feature importance for each model
for model, name in zip(models, model_names):
    plot_feature_importance(model, name)

# Step 5: Test on Unseen Data (Using test set)
X_test = vectorizer.transform(df['Plant Name'])
y_test = df['Watering']
y_pred = watering_model.predict(X_test)
print(f"Accuracy on unseen test data: {accuracy_score(y_test, y_pred)}")

# Step 6: Precision-Recall Curve for all models
def plot_precision_recall_curve(model, X_test, y_test, model_name):
    y_scores = model.predict_proba(X_test)
    for i in range(y_test.nunique()):
        precision, recall, thresholds = precision_recall_curve(y_test == i, y_scores[:, i])
        plt.plot(recall, precision, label=f'Class {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend()
    plt.show()

# Generate Precision-Recall Curve for all models
for model, X, y, name in zip([growth_model, sunlight_model, soil_model, watering_model, fertilization_model],
                             [X_test_growth, X_test_sunlight, X_test_soil, X_test_watering, X_test_fertilization],
                             [y_growth_test, y_sunlight_test, y_soil_test, y_watering_test, y_fertilization_test],
                             model_names):
    plot_precision_recall_curve(model, X, y, name)
