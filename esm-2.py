import os
import torch
import numpy as np
from Bio import SeqIO
import esm

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

input_fasta_adp = 'pep.fasta'
output_npy_adp = 'pep.npy'

def process_fasta_file(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        seq_id = record.id
        seq = str(record.seq)
        sequences.append((seq_id, seq))
    return sequences

def extract_features_and_save(sequences, output_file):
    sequence_features = []
    for seq_id, seq in sequences:
        batch_data = [(seq_id, seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        
        token_representations = results["representations"][33]
        sequence_representation = token_representations[:, 1:-1].mean(0).cpu().numpy()  
        sequence_features.append(sequence_representation)

    sequence_features = np.array(sequence_features)
    np.save(output_file, sequence_features)

sequences_adp = process_fasta_file(input_fasta_adp)
extract_features_and_save(sequences_adp, output_npy_adp)

print("ADP Features saved to:", output_npy_adp)

adp_features = np.load(output_npy_adp)
print("\nFirst 3 ADP sequences' features:")
print(adp_features[:3])

print("\nShapes of the first 3 ADP sequences' features:")
for i in range(3):
    print(f"Sequence {i+1}: {adp_features[i].shape}")
