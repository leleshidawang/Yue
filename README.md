# README
We propose two deep learning-based antidiabetic peptide (ADP) prediction models. One is the CNN model, which has fast prediction speed and high accuracy in 10-fold cross-validation. The other is a three-channel neural network model, including convolutional neural network (CNN), bidirectional long short-term memory network (Bi-LSTM) and recurrent neural network (RNN). This model has the highest accuracy in ten-fold cross-validation.

First of all, for fasta format files, you should ensure that each sequence is 41 amino acids in length, and use 'X' to complete sequences that are insufficient in length. Subsequently, use esm2.py to process the fasta file. Note that please change the input and output file names in esm2.py. The command to run this code is as follows:

python esm2.py

Next, you can choose to use the CNN model or the three-channel neural network model to predict your npy file. The commands to run these codes are as follows:

python CNN.py
or
python CNN_RNN_Bi-LSTM.py

Don't forget to modify the input and output file names in these codes.
