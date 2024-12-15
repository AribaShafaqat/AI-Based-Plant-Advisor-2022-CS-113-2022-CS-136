# Required Libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'E:\\ai\\ai lab conference\\fiiii.csv'  # Update with your dataset path
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Fill missing values
df_filled = df.fillna('Unknown')
# Map Growth column to numeric values
growth_mapping = {"Slow": 1, "Moderate": 2, "Fast": 3}
df["Growth_Num"] = df["Growth"].map(growth_mapping)










# Count the occurrences of each category
growth_counts = df['Growth'].value_counts()

# Plot the pie chart
plt.figure(figsize=(7, 7))
plt.pie(growth_counts, labels=growth_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title("Distribution of Growth Categories")
plt.show()

plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x='Growth', y='Watering', palette='Set2')

# Title and labels
plt.title("Watering Distribution by Growth Categories")
plt.xlabel("Growth Categories")
plt.ylabel("Watering Frequency")
plt.show()

# Plot box plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Growth', y='Watering', palette='Set2')

# Title and labels
plt.title("Watering Distribution by Growth Categories")
plt.xlabel("Growth Categories")
plt.ylabel("Watering Frequency")
plt.show()

# Count the occurrences of each category
growth_counts = df['Growth'].value_counts()

# Plot the frequency distribution of the categories
plt.figure(figsize=(8, 5))
sns.barplot(x=growth_counts.index, y=growth_counts.values, palette='viridis')

# Highlight potential outliers (categories with low frequency)
for i, value in enumerate(growth_counts.values):
    if value < 2:  # Threshold for detecting outliers (adjust as needed)
        plt.text(i, value + 0.1, f'Outlier: {growth_counts.index[i]}', ha='center')

plt.title("Frequency of Growth Categories with Potential Outliers")
plt.xlabel("Growth Categories")
plt.ylabel("Frequency")
plt.show()




category_distribution = df.groupby("Sunlight")["Growth"].value_counts().unstack().fillna(0)

# Plot a bar chart
category_distribution.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="viridis")

# Add labels and title
plt.title("Category Distribution: Plant Growth vs Illumination Conditions")
plt.xlabel("Illumination Condition",fontsize=14)
plt.ylabel("Number of Plants",fontsize=14)
plt.legend(title="Growth", labels=["Slow", "Moderate", "Fast"], loc="upper right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(rotation=0,fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()




# Initialize label encoder
label_encoder = LabelEncoder()

# Encode string/object columns
for col in df_filled.select_dtypes(include=['object']).columns:
    df_filled[col] = label_encoder.fit_transform(df_filled[col])

# Check target distribution and filter classes with only 1 sample
df_filtered = df_filled[df_filled['Growth'] != 4]

# Define features and target
X = df_filtered.drop('Growth', axis=1)  # Features
y = df_filtered['Growth']  # Target

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Define models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=5000, random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "LightGBM": lgb.LGBMClassifier(random_state=42),
   # "CatBoost": CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=10, random_state=42, verbose=0),
}

# Stratified K-Fold Cross Validation
print("Model Performance (Stratified 5-Fold Cross-Validation):")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store results for plotting
model_names = []
model_accuracies = []
model_stds = []

# Evaluate each model
for name, model in models.items():
    accuracies = []
    for train_idx, test_idx in skf.split(X_scaled, y_resampled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_resampled.iloc[train_idx], y_resampled.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        accuracies.append(acc)
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"{name} Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    model_names.append(name)
    model_accuracies.append(mean_accuracy)
    model_stds.append(std_accuracy)
# param_grid = {
#     'num_leaves': [31, 50, 70],
#     'max_depth': [-1, 5, 10],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [100, 500, 1000],
#     'min_data_in_leaf': [10, 20, 30],
#     'feature_fraction': [0.6, 0.8, 1.0]
# }

# # Initialize LightGBM model
# lgb_model = lgb.LGBMClassifier()

# # Grid search
# grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
# grid_search.fit(X_train, y_train)

# # Best parameters and accuracy
# print("Best Parameters:", grid_search.best_params_)
# print("Best Accuracy:", grid_search.best_score_)


# Plot the results
plt.figure(figsize=(12, 6))
plt.barh(model_names, model_accuracies, xerr=model_stds, capsize=5, color='skyblue')
plt.xlabel('Accuracy')
plt.title('Model Comparison')
plt.show()

plt.figure(figsize=(10, 6))

# Adjust the font size for the plot labels
sns.set_context("notebook", font_scale=1.2)  # Increase font size for labels

# Create the pairplot
sns.pairplot(df_filtered, diag_kind='kde', hue='Growth')

# Adjust the title position and increase the font size
plt.suptitle("Scatter Plot Analysis", y=1.02, fontsize=14)

# Show the plot
plt.show()


# # Train-test split (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# # Train Random Forest Classifier
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = rf_model.predict(X_test)

# Generate the classification report specifying labels
# growth_encoder = LabelEncoder()
# growth_encoder.fit(df_filtered['Growth'])
# report = classification_report(y_test, y_pred, target_names=growth_encoder.classes_.astype(str))

# # Print the classification report
# print("Random Forest Classification Report:")
# print(report)

# # Save the classification report to a DataFrame (optional)
# report_dict = classification_report(y_test, y_pred, target_names=growth_encoder.classes_.astype(str), output_dict=True)
# report_df = pd.DataFrame(report_dict).transpose()

# # Save the table to a CSV file (optional)
# report_df.to_csv('classification_report.csv', index=True)

# # Display the DataFrame for reference
# print(report_df)

# Set up the figure and axes for plotting model accuracies
fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1x2 grid for two plots

# Custom function to add legend above each plot
def add_color_legend(ax, column_name, palette):
    unique_values = df[column_name].unique()
    colors = sns.color_palette(palette, len(unique_values))
    handles = [plt.Line2D([0], [0], marker='o', color=colors[i], linestyle='', markersize=8, label=val) 
               for i, val in enumerate(unique_values)]
    ax.legend(handles=handles, title=column_name, bbox_to_anchor=(0.5, 1.3), loc='center', ncol=3, frameon=False)

# Perform One-Hot Encoding for categorical columns (you can list all your categorical columns here)
data_encoded = pd.get_dummies(df, columns=['Sunlight', 'Watering', 'Soil Type', 'Fertilization Tip'], drop_first=True)

# If 'Plant Name' is not useful for clustering, drop it (optional)
data_encoded = data_encoded.drop(columns=['Plant Name'])

# Check the resulting encoded dataframe
print(data_encoded.head())

"""
# from sklearn.cluster import KMeans

# # Apply KMeans clustering to the one-hot encoded data
# kmeans = KMeans(n_clusters=3, random_state=42)
# clusters = kmeans.fit_predict(data_encoded)

# # Add the cluster labels to your dataframe
# data_encoded['Cluster'] = clusters

# # Check the data with cluster labels
# print(data_encoded.head())
"""

# Initialize label encoder
le = LabelEncoder()

# Apply label encoding to Sunlight and Watering columns
df['Sunlight'] = le.fit_transform(df['Sunlight'])
df['Watering'] = le.fit_transform(df['Watering'])

# Optionally, encode 'Growth' if needed for analysis
df['Growth'] = le.fit_transform(df['Growth'])

# Calculate Sunlight-Watering Efficiency Ratio
df['Sunlight-Watering Ratio'] = df['Sunlight'] / df['Watering']

# Generate a unique color for each unique plant dynamically
unique_plants = df['Plant Name'].unique()
cmap = plt.cm.get_cmap('hsv', len(unique_plants))  # Dynamically generate colors
plant_colors = {plant: cmap(i) for i, plant in enumerate(unique_plants)}

# Plotting the graph
plt.figure(figsize=(10, 8))  # Increased figure size

# Plot each plant with its unique color
for i, plant in enumerate(df['Plant Name']):
    plt.scatter(
        df['Sunlight-Watering Ratio'][i], 
        df['Growth'][i], 
        color=plant_colors[plant], 
        label=plant, 
        s=100  # Marker size
    )

# Label the plot
plt.title('Sunlight-Watering Efficiency Ratio vs. Plant Growth', fontsize=14)
plt.xlabel('Sunlight-Watering Efficiency Ratio', fontsize=12)
plt.ylabel('Growth', fontsize=12)

# Add a legend to show which color represents which plant
plt.legend(title="Plant Names", loc='best', fontsize=10)

# Display the plot
plt.grid(True)
plt.show()
# Plot 1: Distribution of Growth
sns.countplot(x='Growth', data=df, palette='Set2', ax=axs[0])
axs[0].set_title('Distribution of Growth')
axs[0].set_ylabel('Count')
axs[0].tick_params(axis='x', bottom=False, labelbottom=False)  # Hide x-axis labels and ticks
add_color_legend(axs[0], 'Growth', 'Set2')

# Plot 2: Distribution of Sunlight
sns.countplot(x='Sunlight', data=df, palette='Set2', ax=axs[1])
axs[1].set_title('Distribution of Sunlight Requirements')
axs[1].set_ylabel('Count')
axs[1].tick_params(axis='x', bottom=False, labelbottom=False)  # Hide x-axis labels and ticks
add_color_legend(axs[1], 'Sunlight', 'Set2')

# Adjust layout
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

# Set the color palette to a more distinct one
sns.set_palette("Set2")

# Create the bar plot
sns.barplot(x='Sunlight', y='Growth', hue='Watering', data=df, ci=None, dodge=True)

# Enhance the chart with clear labels and a title
plt.title('Plant Growth Under Different Sunlight and Watering Conditions', fontsize=16)
plt.xlabel('Sunlight Conditions', fontsize=12)
plt.ylabel('Growth (in cm)', fontsize=12)

# Rotate x-axis labels for readability
plt.xticks(rotation=90, ha='right')

# Add a legend for the watering categories
plt.legend(title='Watering Frequency', loc='upper left', bbox_to_anchor=(1, 1))

# Improve layout and display the plot
plt.tight_layout()
plt.show()

# Optional: Plotting the results for better visualization
# Generate random colors for each bar
colors = np.random.rand(len(model_names), 3)  # Random colors (RGB)

# Plotting the accuracies of different models
plt.figure(figsize=(10, 6))
plt.barh(model_names, model_accuracies, xerr=model_stds, capsize=5, color=colors)
plt.xlabel('Accuracy')
plt.title('Model Comparison (Stratified 5-Fold Cross-Validation)')
plt.show()

# Plot for Growth
plt.figure(figsize=(8,6))
sns.countplot(x='Growth', data=df, palette='Set2')
plt.title('Distribution of Plant Growth')
plt.xlabel('Growth Rate')
plt.ylabel('Count')
plt.show()

# Plot for Sunlight Requirements
plt.figure(figsize=(8,6))
sns.countplot(x='Sunlight', data=df, palette='Set2')
plt.title('Distribution of Sunlight Requirements')
plt.xlabel('Sunlight Condition')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Plot for Watering Frequency
plt.figure(figsize=(10,6))
sns.countplot(x='Watering', data=df, palette='Set2')
plt.title('Distribution of Watering Frequency')
plt.xlabel('Watering Frequency')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Plot for Soil Type
plt.figure(figsize=(10,6))
sns.countplot(x='Soil Type', data=df, palette='Set2')
plt.title('Distribution of Soil Types')
plt.xlabel('Soil Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

from wordcloud import WordCloud

# Combine all fertilization tips into one string
text = ' '.join(df['Fertilization Tip'].dropna())

# Create the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Plot the word cloud
plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Fertilization Tips')
plt.axis('off')
plt.show()





# Set up the figure and subplots with a larger figure size
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Bar plot for Sunlight Requirements
sns.countplot(x='Sunlight', data=df, ax=axs[0, 0], palette='Set2')
axs[0, 0].set_title('Bar Plot for Sunlight Requirements')
axs[0, 0].set_xlabel('Sunlight')
axs[0, 0].set_ylabel('Count')
axs[0, 0].tick_params(axis='x', rotation=90)  # Rotate x-axis labels

# Bar plot for Watering Frequency
sns.countplot(x='Watering', data=df, ax=axs[0, 1], palette='Set3')
axs[0, 1].set_title('Bar Plot for Watering Frequency')
axs[0, 1].set_xlabel('Watering Frequency')
axs[0, 1].set_ylabel('Count')
axs[0, 1].tick_params(axis='x', rotation=90)  # Rotate x-axis labels

# Bar plot for Growth
sns.countplot(x='Growth', data=df, ax=axs[1, 0], palette='Set1')
axs[1, 0].set_title('Bar Plot for Growth')
axs[1, 0].set_xlabel('Growth Speed')
axs[1, 0].set_ylabel('Count')
axs[1, 0].tick_params(axis='x', rotation=90)  # Rotate x-axis labels

# Bar plot for Soil Type
sns.countplot(x='Soil Type', data=df, ax=axs[1, 1], palette='Paired')
axs[1, 1].set_title('Bar Plot for Soil Type')
axs[1, 1].set_xlabel('Soil Type')
axs[1, 1].set_ylabel('Count')
axs[1, 1].tick_params(axis='x', rotation=90)  # Rotate x-axis labels

# Adjust layout for better spacing and avoid overlap
plt.tight_layout()

# Show the plots
plt.show()



plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Growth', hue='Sunlight', palette='coolwarm')
plt.title('Bar Chart of Plant Growth vs. Sunlight Type')
plt.xlabel('Growth Speed')
plt.ylabel('Count')
plt.legend(title='Sunlight Type')
plt.show()

# Automatically encode categorical columns
df['Growth (Encoded)'] = pd.Categorical(df['Growth']).codes + 1
df['Sunlight (Encoded)'] = pd.Categorical(df['Sunlight']).codes + 1
df['Watering (Encoded)'] = pd.Categorical(df['Watering']).codes + 1

# Scatter plot with bold dots and distinct colors
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=df,
    x='Sunlight (Encoded)',
    y='Watering (Encoded)',
    hue='Growth',  # Use colors to differentiate growth speeds
    s=200,  # Larger size for bold dots
    palette='tab10',  # Professional color palette
    edgecolor='black',  # Black border around dots for contrast
    linewidth=0.8  # Slight border thickness
)

# Customize plot
plt.title('Scatter Plot: Sunlight vs Watering Frequency by Growth Speed', fontsize=14, fontweight='bold')
plt.xlabel('Sunlight Requirements (Encoded)', fontsize=12)
plt.ylabel('Watering Frequency (Encoded)', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Growth Speed', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Customize plot
plt.title('Scatter Plot: Sunlight vs Watering Frequency by Growth Speed', fontsize=14)
plt.xlabel('Sunlight Requirements (Encoded)', fontsize=12)
plt.ylabel('Watering Frequency (Encoded)', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Growth Speed', fontsize=10)

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Correlation Analysis & Heatmap
plt.figure(figsize=(10, 6))
corr_matrix = df_filtered.corr()  # Correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Scatter Plot Analysis for relationship between features
plt.figure(figsize=(10, 6))
sns.pairplot(df_filtered, diag_kind='kde', hue='Growth')  # Scatter matrix with KDE on diagonals
plt.suptitle("Scatter Plot Analysis", y=1.02)
plt.show()


# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot Growth vs Sunlight
growth_vs_sunlight = df_filtered.groupby('Sunlight')['Growth'].mean()
sns.lineplot(x=growth_vs_sunlight.index, y=growth_vs_sunlight.values, marker='o', color='red', ax=axs[0])
axs[0].set_title('Growth vs Sunlight', fontsize=16)
axs[0].set_xlabel('Sunlight', fontsize=14)
axs[0].set_ylabel('Average Growth', fontsize=14)
axs[0].grid(True)
axs[0].tick_params(axis='x', rotation=0, labelsize=12)
axs[0].tick_params(axis='y', labelsize=12)

# Plot Growth vs Watering
growth_vs_watering = df_filtered.groupby('Watering')['Growth'].mean()
sns.lineplot(x=growth_vs_watering.index, y=growth_vs_watering.values, marker='o', color='blue', ax=axs[1])
axs[1].set_title('Growth vs Watering', fontsize=16)
axs[1].set_xlabel('Watering Frequency', fontsize=14)
axs[1].set_ylabel('Average Growth', fontsize=14)
axs[1].grid(True)
axs[1].tick_params(axis='x', rotation=0, labelsize=12)
axs[1].tick_params(axis='y', labelsize=12)

# Plot Growth vs Soil Type
growth_vs_soil = df_filtered.groupby('Soil Type')['Growth'].mean()
sns.lineplot(x=growth_vs_soil.index, y=growth_vs_soil.values, marker='o', color='black', ax=axs[2])
axs[2].set_title('Growth vs Soil Type', fontsize=16)
axs[2].set_xlabel('Soil Type', fontsize=14)
axs[2].set_ylabel('Average Growth', fontsize=14)
axs[2].grid(True)
axs[2].tick_params(axis='x', rotation=0, labelsize=13)
axs[2].tick_params(axis='y', labelsize=13)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()



# # Outlier Detection using Z-Score
# z_scores = np.abs(stats.zscore(X_scaled))
# outliers = (z_scores > 3).all(axis=1)  # Threshold of 3 for outliers
# print(f"Number of outliers detected: {np.sum(outliers)}")
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(X_scaled)), X_scaled[:, 0], color='blue', label='Normal')
# plt.scatter(np.where(outliers)[0], X_scaled[outliers, 0], color='red', label='Outliers')
# plt.title("Outlier Detection")
# plt.legend()
# plt.show()

# Skewness Plot
# Calculate skewness for each feature in the dataset
skewness = df_scaled = pd.DataFrame(X_scaled, columns=X.columns).skew()

# Plot the skewness of each feature
plt.figure(figsize=(10, 6))
sns.barplot(x=skewness.index, y=skewness.values, color='skyblue')
plt.title("Skewness of Each Feature")
plt.xlabel('Feature')
plt.ylabel('Skewness')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

# Log Transformation of Growth (or any other column you want to transform)
df_filled['Growth_encoded'] = label_encoder.fit_transform(df_filled['Growth'])  # Encode 'Growth' first

# Apply log transformation to 'Growth_encoded' (adding 1 to avoid log(0) issues)
df_transformed = df_filled.copy()
df_transformed['Growth_encoded'] = np.log1p(df_transformed['Growth_encoded'])

# Set up the plotting grid (1 row, 2 columns for the original vs log-transformed distributions)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot original and log-transformed distributions for Growth
sns.histplot(df_filled['Growth_encoded'], kde=True, color='blue', label='Original', ax=axes[0])
sns.histplot(df_transformed['Growth_encoded'], kde=True, color='red', label='Log-Transformed', ax=axes[1])

# Set titles and labels
axes[0].set_title('Original Growth Distribution')
axes[0].set_xlabel('Growth (Encoded)')
axes[0].set_ylabel('Density')
axes[0].legend()

axes[1].set_title('Log-Transformed Growth Distribution')
axes[1].set_xlabel('Log(1 + Growth_encoded)')
axes[1].set_ylabel('Density')
axes[1].legend()

plt.tight_layout()
plt.show()
