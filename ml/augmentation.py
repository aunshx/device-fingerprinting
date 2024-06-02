import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# Load data from JSON file
with open('./data/extracted_component_data.json') as file:
    data = json.load(file)

# Prepare data for the model
platforms = []
features = []

def augment_operation_output(operation_output, num_augmentations=5):
    augmented_outputs = []
    for _ in range(num_augmentations):
        noise = np.random.normal(0, 0.1, len(operation_output))
        scaled = operation_output * np.random.uniform(0.9, 1.1)
        shifted = operation_output + np.random.uniform(-0.1, 0.1)
        augmented_outputs.append(operation_output + noise)
        augmented_outputs.append(scaled)
        augmented_outputs.append(shifted)
    return augmented_outputs

for entry in data:
    platform = entry['components']['platform']['value']
    operation_output_str = entry['operationOutput'].strip('[]').split(', ')
    
    # Convert operationOutput to numerical values
    try:
        operation_output = np.array(list(map(float, operation_output_str)))
    except ValueError as e:
        print(f"Error converting operationOutput to float: {e}")
        continue
    
    # Augment the data
    augmented_outputs = augment_operation_output(operation_output)
    
    for augmented_output in augmented_outputs:
        # Extract statistical features from operationOutput
        feature = [
            np.mean(augmented_output),
            np.median(augmented_output),
            np.std(augmented_output),
            np.min(augmented_output),
            np.max(augmented_output)
        ]
        platforms.append(platform)
        features.append(feature)

# Create DataFrame
df = pd.DataFrame(features, columns=['mean', 'median', 'std', 'min', 'max'])
df['platform'] = platforms

# Encode target variable
platform_categories = df['platform'].astype('category').cat.categories
df['platform'] = df['platform'].astype('category').cat.codes

# Split data into training and testing sets
X = df.drop('platform', axis=1)
y = df['platform']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Function to predict platform from operationOutput with probabilities
def predict_platform_with_prob(operation_output):
    feature = [
        np.mean(operation_output),
        np.median(operation_output),
        np.std(operation_output),
        np.min(operation_output),
        np.max(operation_output)
    ]
    probabilities = clf.predict_proba([feature])[0]
    predicted_index = np.argmax(probabilities)
    predicted_platform = platform_categories[predicted_index]
    probability_percentage = probabilities[predicted_index] * 100
    return predicted_platform, probability_percentage, probabilities

# Example usage
example_output_str = "[9.934999980032444, 18.99999985471368, 20.594999892637134, 62.66500009223819, 73.33999988622963, 64.61500003933907, 27.024999959394336, 73.87500000186265, 82.84500008448958, 60.870000161230564, 94.56499991938472, 68.78499989397824, 85.28500003740191, 43.01999998278916, 7.044999860227108, 54.04500011354685, 72.07999983802438, 6.275000050663948, 34.075000090524554, 28.00000016577542, 5.304999882355332, 21.42000012099743, 8.02999990992248, 52.86499997600913, 54.34500006958842, 83.42999988235533, 93.43999996781349, 41.130000026896596, 66.02000002749264, 64.08500019460917, 49.80999999679625, 32.06500015221536, 228.815000038594, 33.1500000320375, 98.29499991610646, 51.310000009834766, 44.90999993868172, 55.86500000208616, 41.80000000633299, 115.25500006973743, 12.539999792352319, 22.210000082850456, 116.9100000988692, 74.45499999448657, 34.06500001437962, 31.984999775886536, 125.56500011123717, 33.12999987974763, 32.7050001360476, 66.64500012993813, 75.26999991387129, 65.75500010512769, 20.980000030249357, 57.4949998408556, 73.28999997116625, 32.1450000628829, 36.124999867752194, 83.97500007413328, 56.55999993905425, 17.47000007890165, 61.074999859556556, 50.4400001373142, 55.49499997869134, 80.49500011838973, 49.739999929443, 781.8400000687689, 50.909999990835786, 53.86499990709126, 65.51000010222197, 77.36500003375113, 128.7599999923259, 46.925000147894025, 45.26999988593161, 165.49500008113682, 20.164999878033996, 10.565000120550394, 11.414999840781093, 14.565000077709556, 34.34500005096197, 65.53000002168119, 35.18000012263656, 21.595000056549907, 8.700000122189522, 23.090000031515956, 67.73000000976026, 33.94500003196299, 64.76500001735985, 69.01999982073903, 16.70500007458031, 6.34000007994473, 27.279999805614352, 142.3599999397993, 42.52499993890524, 86.47500001825392, 30.15500004403293, 91.74000006169081, 78.34999985061586, 70.20499999634922, 22.30999991297722, 17.72500015795231, 29.45000003091991, 5.690000019967556, 118.55500005185604, 61.944999964907765, 18.11000006273389, 89.5450001116842, 45.33999995328486, 66.42500008456409, 31.334999948740005, 46.799999894574285, 73.6950000282377, 51.79499997757375, 72.5749998819083, 89.91000009700656, 229.32499996386468, 111.55000003054738, 104.96999998576939, 56.77999998442829, 66.65000016801059, 166.39499994926155, 177.96000000089407, 149.1749999113381, 182.70000000484288, 155.399999814108, 200.58000017888844, 87.1099999640137, 127.76000006124377, 151.48000000044703, 83.76499987207353, 63.22999997064471, 65.66000008024275, 1997.5299998186529, 173.15500020049512, 121.34500010870397, 260.60499995946884, 317.3999998252839, 209.80000006966293, 400.74499999172986, 169.19000004418194, 103.42999990098178, 83.64999992772937, 206.32999995723367, 72.14000006206334, 69.70499991439283, 120.36999990232289, 129.7400000039488, 142.25500007160008, 194.88500012084842, 94.9649999383837, 82.054999]"

example_output = np.array(list(map(float, example_output_str.strip('[]').split(', '))))

predicted_platform, probability, probabilities = predict_platform_with_prob(example_output)
print(f'Predicted Platform: {predicted_platform} Probability: {probability:.2f}%')

# To print probabilities for all classes
for i, prob in enumerate(probabilities):
    print(f'Platform: {platform_categories[i]} Probability: {prob * 100:.2f}%')
