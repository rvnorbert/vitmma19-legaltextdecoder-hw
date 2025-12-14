# Deep Learning Class (VITMMA19) Project: Legal Text Decoder

## Project Details

### Project Information

- **Selected Topic**: Legal Text Decoder
- **Student Name**: Árva Norbert Ákos
- **Aiming for +1 Mark**: No

### Solution Description

I implemented a solution using TensorFlow 2.16.1 to figure out the readability of Hungarian legal texts (ASZF) on a 1-to-5 scale.

**Methodology:**
1.  **Data Processing**: JSON files from Label Studio are processed, extracting text and labels. The dataset is split into Train (70%), Validation (15%), and Test (15%) sets.
2.  **Model Architecture**: I used a pre-trained **`distilbert-base-multilingual-cased`** model as the backbone to generate sentence embeddings. To optimize for memory, as my GPU is not the strongest out there (4GB VRAM limit), and performance:
    * The DistilBERT layers are **frozen** (non-trainable).
    * A custom **Keras Functional API** head was added: A Dense layer (64 units, ReLU) followed by Dropout (0.2) and a final Output layer (5 units).
3.  **Training**: The model is trained using the **Adam** optimizer and Sparse Categorical Crossentropy loss. I employed **Class Weighting** to handle imbalanced data and **Early Stopping** to prevent overfitting.
4.  **Performance**: The model is evaluated using Accuracy and a Confusion Matrix on the hold-out test set.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t legal-decoder .
```

#### Run

To run the solution, use the following command. 
This assumes that your current data, used for training, validation and testing, is under `/data` in the directory from which the script is executed .

This mounts your local `./data` directory to `/app/data` inside the container, and the `./output` directory to `/app/output` inside the container.

To capture the logs, the output is redirected to a file:

```bash
# PowerShell example
docker run --rm --gpus all `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/output:/app/output `
  legal-decoder > log/run.log 2>&1
```
* Ensure your JSON data files are in the local `./data` folder.
* The output (models, reports, logs) will be saved to the `./output` folder.
* Ensure the log folder exists
* The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.

### Data Preparation

The system expects raw JSON files exported from Label Studio, formatted according to the Legal Text Decoder specifications, placed in the `/data` directory.
The script `src/01_data_preprocessing.py` will automatically:

1.  Recursively finds all `*.json` files in the data directory.
2.  Parses the JSON structure to extract the `text` field and the `label` (from the annotation result).
3.  Converts labels (e.g., "3-Average") to integer format (0-4).
4.  Splits the data into `train.csv`, `val.csv`, and `test.csv` and saves them to the output directory.

### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01_data_processing.py`: Scripts for loading raw JSON data, cleaning text, and splitting into train/val/test sets.
    - `02_train.py`: The main script for defining the model and executing the training loop.
    - `03_evaluation.py`: Scripts for evaluating the trained model on test data and generating metrics.
    - `04_inference.py`: Script for running the model on new, unseen data to generate predictions.
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs, batch size) and paths.
    - `utils.py`: Helper functions and utilities used across different scripts.
    - `run.sh`: Shell script used as the entry point for the pipeline execution.

- **`notebook/`**: Directory for Jupyter notebooks used for experiments and analysis.

- **`log/`**: Directory for storing execution logs.
  - `run.log`: Log file showing the output a run.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.

## Differences

### Difference 1

**Alternative to the given PowerShell run command**

Alternative to the powershell run command given above is this windows cmd command, where you replace `<PATH_TO_DATA>` with the absolute path to the local data folder and `<PATH_TO_OUTPUT>` with the absolute path to the local folder where you want the output to be placed.

**Windows (Command Prompt):**
```cmd
docker run --rm --gpus all ^
  -v "<PATH_TO_DATA>":/app/data ^
  -v "<PATH_TO_OUTPUT>":/app/output ^
  legal-decoder > log/run.log 2>&1
```

### Difference 2

**Possible change in run command if folder structure inside the container does not meet the requirements**

The 2 given sources for the example run command, namely the "Projekt munka részletes ismertető: Legal Text Decoder" and the file vitmma19-pw-template/README.md inside the template repository have different structures for the `/data` folder inside the container.

If the run command does not work, change the mounting from `:/app/data` to `:/data`. Example with the original **PowerShell** script

```bash
# PowerShell example
docker run --rm --gpus all `
  -v ${PWD}/data:/data `
  -v ${PWD}/output:/app/output `
  legal-decoder > log/run.log 2>&1
```




