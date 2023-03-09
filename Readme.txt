Instruction:
This model using CNN algorithm and a customized layer :L1Dist to create the Siamese_model
Through learning the similarities between anchor and positive (me and myself) and the differences between anchor and negatives(me and all other people)
Complete the verification though comparing the differences between new_inputimage and positive/negative class: will return True or False
Could imporve the accuracy thourgh : increase threshold, increase learning rate(not necessary in this case), increase sample size, increase number of epchos


To run this app:
First install package from requirements(optional: if possible use GPU to run tensorflow)
Run trainmodel to collect positive and anchor input
After training model from modeltrain file, save and load it into the same folder
run faceid_app file with layers and model.h5 in the same folder.
Using the pop-up camera to verify yourself

This project follow the insturctions by:
https://www.youtube.com/watch?v=LKispFFQ5GU
Other Reference Link:
model Paper: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf (Siamese_model)
Labelled Faces in the Wild: http://vis-www.cs.umass.edu/lfw/ (data from negative class)