import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import spacy
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Load the dataset
data = pd.read_csv('ok.csv', encoding='latin1')

# Strip extra whitespace from column names
data.columns = data.columns.str.strip()

# Check for missing values
print("Missing Values:\n", data.isnull().sum())

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Check unique values in relevant columns
print("Unique values in Growth column:", data['Growth'].unique())
print("Unique values in Soil Type column:", data['Soil'].unique())
print("Unique values in Sunlight column:", data['Sunlight'].unique())

# Encode the Plant Name column (if it's categorical)
if 'Plant Name' in data.columns:
    data['Plant Name'] = label_encoder.fit_transform(data['Plant Name'])

# Encode other categorical columns
data['Growth'] = label_encoder.fit_transform(data['Growth'])
data['Soil'] = label_encoder.fit_transform(data['Soil'])
data['Sunlight'] = label_encoder.fit_transform(data['Sunlight'])

# Display the processed dataset
print("\nProcessed Dataset:\n", data.head())

# Standard scaling (apply only to relevant numerical features, not encoded ones)
scaler = StandardScaler()
# You can choose to scale numerical columns if necessary; skip scaling for categorical columns
data[['Growth']] = scaler.fit_transform(data[['Growth']])  # If scaling is necessary
data[['Soil']] = scaler.fit_transform(data[['Soil']])      # If scaling is necessary
data[['Sunlight']] = scaler.fit_transform(data[['Sunlight']])  # If scaling is necessary

print("\nProcessed Dataset with Scaling:\n", data)

# Function to dynamically extract conditions from sentences
def extract_watering_info(sentence):
    # Process the sentence using spaCy NLP pipeline
    doc = nlp(sentence)
    
    # Define empty dictionary to store extracted information
    info = {
        'frequency': None,
        'condition': None
    }

    # Check for frequency (e.g., "once a week", "every 2 days")
    frequency_pattern = re.search(r'(every|once|twice)\s(\d+)?\s*(days|weeks|day|week)', sentence, re.IGNORECASE)
    if frequency_pattern:
        info['frequency'] = frequency_pattern.group(2) or '1'  # Extract number or default to '1'

    # Handle non-numeric frequencies
    elif "once a week" in sentence.lower():
        info['frequency'] = '1'
    elif "twice a week" in sentence.lower():
        info['frequency'] = '2'
    elif "regular" in sentence.lower():
        info['frequency'] = 'regular'

    # Detect soil conditions
    soil_conditions = ["moist", "dry", "slightly moist", "topsoil dry", "proper drainage", "drainage", "feels dry"]
    for condition in soil_conditions:
        if condition in sentence.lower():
            info['condition'] = condition
            break  # Stop at the first matched condition

    return info

# Load spaCy model for English
nlp = spacy.load("en_core_web_sm")

# Apply the function to the dataset if 'Watering' column exists
if 'Watering' in data.columns:
    data['watering_info'] = data['Watering'].apply(extract_watering_info)
    # Check the result
    print("\nWatering Info:\n", data[['Watering', 'watering_info']])
else:
    print("No 'Watering' column found in the dataset.")

# Handle missing values by dropping rows with NaN values (or fill with some value)
data = data.dropna()

# Split the data into features (X) and targets (y)
X = data.drop(['Growth', 'Soil', 'Watering', 'Sunlight'], axis=1)  # Keep 'Plant Name' and other features
y = data[['Growth', 'Soil', 'Watering', 'Sunlight']]  # These are the target features

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model for multi-output classification
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100))

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nOverall Accuracy for multi-output classification:", accuracy)

# Detailed classification report for each target column
print("\nClassification Report for multi-output classification:\n", classification_report(y_test, y_pred))

# Set up hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Print best parameters from GridSearchCV
print("\nBest Hyperparameters:", grid_search.best_params_)
