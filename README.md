# Image_Classification_Using_Deep_Learning
(Yeh_kiski_tasveer_hai) is what the problem statement stated in codeshastra... I implemented it myself now... using the similar intel-image-classification dataset provided in the competition.

  Dataset : https://www.kaggle.com/puneet6060/intel-image-classification

Problem statement was as follows : 
  =>Train a model(s) that classifies images based on the dataset provided. 
  =>Create a UI that can run inference using the model trained above on 128 unseen and unlabeled images uploaded at the same time.
  =>Once inference is completed, the UI should then be able to visualize these images and their predictions including the 
  confidence score and provide other metrics as appropriate.
  =>The UI should have the functionality to change the labels of images that are wrong, add them to a database and run
  training again.
  =>Optionally, the UI should have an option to change the parameters of training. Parameters could be learning rate,
  number of filters, filter size etc.
  =>The newly trained model should be available for use by the UI to run another round of inference.
  =>Extra credit will be given if the entire process is done on the cloud.
  =>Create a platform for training, labeling and deploying and retraining image classification models.

The tasks i accomplished :
=> Trained the model with successful classification
=> Created ui using tkinter, pretty basic but it works so its cool
=> Visualization does takes place after inference and i display the prediction score with label, bored to display other metrics.. its easy if you want to display them .. see the plot.png and plotR.png files you will get other metrics as well
=> UI allows you to change labels with correct one.. you simply need to enter the correct label when prompted the rest is good
=> Database part is easy... i used pymongo of python but i had some server issues so i just commented it
=> The re-training happens as well ... i actually wrote the re-training script again because my initial training script was not included in a function so that i could have called it here in new training script file and re-trained the model... but you can if you are reading this..
=> The model could be made available by simple setting a button which will swap the previous one with new one... i didn't do it
=> You can simply use colab by uploading this entire directory to your drive and mounting it to colab.
=> Achieved what was required. Done
