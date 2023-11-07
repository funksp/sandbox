import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function to detect anomalies and visualize them
def visualize_anomalies(patient_id, data, model, code_to_description, probability_threshold=0.01, use_pca=False):
    # Filter data for the specific patient
    patient_data = data[data['patient_id'] == patient_id]

    # Group the data by claim_id and aggregate the ICD codes into lists
    grouped = patient_data.groupby('claim_id')['icd9_code'].apply(list)

    # Find anomalies in the sequences
    anomalies = []
    for claim_id, sequence in grouped.items():
        if len(sequence) > 1:  # Only consider sequences with more than one code
            for i in range(len(sequence) - 1):
                input_context = sequence[:i + 1]
                predictions = model.predict_output_word(input_context, topn=len(model.wv.index_to_key))

                # Create a dictionary of predicted words and their probabilities
                predictions_dict = {predicted[0]: predicted[1] for predicted in predictions}

                actual_next_code = sequence[i + 1]
                actual_next_code_probability = predictions_dict.get(actual_next_code, 0)

                if actual_next_code_probability < probability_threshold:
                    anomalies.append((claim_id, sequence, i + 1, actual_next_code, actual_next_code_probability))

    # Print the claims with anomalies
    for anomaly in anomalies:
        claim_id, sequence, position, anomalous_code, probability = anomaly
        print(f"Anomaly in patient {patient_id}, claim {claim_id}, position {position}, code {anomalous_code} with probability {probability}")
        for code in sequence:
            description = code_to_description.get(code, "Description not found")
            print(f"{code}: {description}")
        print("\n")

    # Prepare the data for visualization
    all_codes = [code for sequence in grouped.values() for code in sequence]
    vectors = [model.wv[code] for code in all_codes if code in model.wv]
    labels = [code_to_description.get(code, code) for code in all_codes]

    # Run PCA or t-SNE
    if use_pca:
        pca = PCA(n_components=2)
        result = pca.fit_transform(vectors)
    else:
        tsne = TSNE(n_components=2, random_state=0)
        result = tsne.fit_transform(vectors)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.scatter(result[:, 0], result[:, 1], c='blue', edgecolors='k', s=50)

    # Annotate points
    for idx, (label, vector) in enumerate(zip(labels, result)):
        if all_codes[idx] in [anom[3] for anom in anomalies]:
            plt.annotate(label, xy=(vector[0], vector[1]), fontweight='bold')
        else:
            plt.annotate(label, xy=(vector[0], vector[1]))

    plt.show()

# To use this function:
# visualize_anomalies(patient_id, data, model, code_to_description, probability_threshold=0.01, use_pca=False)
