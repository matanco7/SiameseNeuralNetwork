# SiameseNeuralNetwork

This work aims  to achieve two goals:

Successfully implement a Siamese neural network, a form of metric learning with a contrastive triplet loss function. Lastly, to relate the representations to landmarks of interest by k-means clustering.

The data set contains a large private collection of OCT scans and annotations of cyst type and size, divided to triplets of frames (anchor, positive and negative).

The model was implemented in PyTorch from scratch and was inspired by the contracting path of the U-net, reaching 91% accuracy on the validation set from two final networks of 64 and 128 output size, and a different number of output features for k-means clustering.
