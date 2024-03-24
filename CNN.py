from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import numpy as np
import csv

adp_features = np.load('Training_ADP_datasets.npy')
non_adp_features = np.load('Training_non-ADP_datasets.npy')

adp_labels = np.ones(adp_features.shape[0])
non_adp_labels = np.zeros(non_adp_features.shape[0])

np.random.seed(42)
np.random.shuffle(adp_features)
np.random.seed(42)
np.random.shuffle(adp_labels)

np.random.seed(42) 
np.random.shuffle(non_adp_features)
np.random.seed(42) 
np.random.shuffle(non_adp_labels)

num_splits = 10
non_adp_splits = np.array_split(non_adp_features, num_splits)

X_list = [np.concatenate([adp_features, split], axis=0) for split in non_adp_splits]
y_list = [np.concatenate([adp_labels, np.zeros(split.shape[0])], axis=0) for split in non_adp_splits]

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(128, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

with open('predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Dataset', 'Sequence', 'Prediction'])
    
    for i, (X, y) in enumerate(zip(X_list, y_list)):
        print(f"Training and evaluating model for dataset {i+1}")
        model = create_cnn_model(X.shape[1:])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, epochs=28, batch_size=32, verbose=1)
        
        new_data = np.load('adp_independent_testing_datasets.npy')
        predictions = model.predict(new_data)
        
        for j, pred in enumerate(predictions):
            writer.writerow([f'Dataset {i+1}', f'Sequence {j+1}', pred[0]])

