
# Local .py Files

To run the code, you need to install the libraries listed in the "src/requirements.txt" file. To use a GPU, you must install the drivers for CUDA usage. Otherwise, you can use the CPU, but this will significantly increase the execution time.

## Training a Model:

- Place images of diseased leaves in the "data/dataset_training/Diseased" directory and images of healthy leaves in the "data/dataset_training/Healthy" directory.
- Run the function `train_model([model_name])`, where `[model_name]` should be replaced with the name you want to assign to the model (e.g., "resnet50_test").
- After executing the code, the model will be saved in the "models" folder, and the training statistics will be saved in the "stats" folder.

## Making Predictions:

- Place in the "data/dataset_prediction" directory the images for which you want to obtain a prediction.
- Execute the function `get_prediction([model_name])`, where `[model_name]` should be replaced with the name of the model you want to use (e.g., "resnet50_test"). The model must be present in the "data/models" folder.
- After executing the code, in the "outputs" folder, there will be a file named [model name]_output.csv containing the predictions for the images. The "IMAGE" column contains the name of the image, and the "PREDICTION" column contains the class estimated by the model.

### Example Execution from Terminal:

- While in the "src" directory, use:

\```bash
python -c 'import racemus_resnet_grape_classifier; racemus_resnet_grape_classifier.train_model("test")'
\```

# With COLAB Notebooks

- Copy the shared folder "racemus_classifier" to your Google Drive.
- Open the colab notebook "racemus_classifier.ipynb" in "racemus_classifier/src/".
- Go to "Runtime -> Change runtime type" and under "Hardware accelerator" choose "GPU" and save.
- Change the "PATH_TO_FOLDER" variable to the path in your Drive to the "src" folder inside the "racemus_classifier" folder you copied.
- In the "MODEL_NAME" variable, you can change the name to assign to the model for training/using for predictions.
- For quick execution with this method, it is necessary to have the datasets structured as described for the .py file method, but it is also necessary to put them into a zip. Then upload such zips to Drive and indicate in the two "!unzip" commands (in the third cell of the notebook) their paths.
- Statistics/models/outputs will be saved with the same logic indicated for the method with the .py file.

## Execution:

- Run all cells except the last two.
- Run the penultimate cell if you want to train the model with the name indicated in "MODEL_NAME".
- Execute the last cell to make predictions with the model with the name indicated in "MODEL_NAME".
