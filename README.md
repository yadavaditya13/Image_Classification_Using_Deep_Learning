# Image_Classification_With_UI_Using_Deep_Learning
(Yeh_kiski_tasveer_hai) is what the problem statement stated in codeshastra... I implemented it myself this quarentine... using the similar intel-image-classification dataset provided in the competition.

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

The Flow of my Project :
  Initially created a CNN using Keras which will be trained using the datasets... Now I placed it as a package in "my_model" dir.
  This model is then called in the "model_train_script.ipynb" which gives me my trained model and its label encodings as "jagga.model" and "le.pickle" respectively...
  I created another package called "model_retrain" which stores the script for retraining my model
  Further we have the final script that will be run by the user called "image_classifier.py"...
  This script has basic ui which allows user to select 128 images anonymously from the asked dir of dataset... Further once the run_inference command is activated they will be able to see the results and have three options :
  1) Continue without interrupting flow by clicking "c" from keyboard.
  2) Renaming the wrong label by giving keyboard interrupt of "r".
  3) Exiting from the process by giving interrupt "q".
  
  Once inference is accomplished... They will be prompted to retrain the model and then they will be asked to give desired parameters such as learning_rate, batch_size and epoch #...
  Once submitted... Retraining will begin and after finishing this process the retrained model will replace the existing "jagga.model" so that it can be used for further inference running...
  Also as asked I am storing the new tested images with their modeified labels in MongoDB using pymongo... And also saving a copy in train dir.
  Now one big advantage for user i gave is that if he gives a new label which machine has no idea of it will still run perfectly fine during retraining because my script will save it in training datset directory under the given label...So i would say Everything asked for is finally accomplished. And yes this will run on cloud by simply uploading it to colab and just run it.
