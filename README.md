# PoliticalBiasChecker
Sentiment analysis tool that identifies political bias within media (specifically twitter posts)

## Project Setup

This project requires Python 3.x and the necessary dependencies installed. To ensure all dependencies are available, it's recommended to create a virtual environment and install required packages.

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Installation

1. **Clone the Repository**

```bash
git clone <repository-url>
cd <project-directory>
```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

To run the project, you need to preprocess the data for training and testing, and then execute the main script. Follow the steps below.

### Step 1: Preprocess Training Data

Run the following command to preprocess the training data:

```bash
python preprocess_train.py
```

This script processes the training dataset and prepares it for the main execution.

---

### Step 2: Preprocess Testing Data

Run the following command to preprocess the testing data:

```bash
python preprocess_test.py
```

This script processes the testing dataset to ensure it is ready for the main program.

---

### Step 3: Run the Main Program

Once the training and testing data have been preprocessed, you can run the main program using:

```bash
python main.py
```

This will execute the core logic of the project and utilize the preprocessed data files.

---

## Additional Notes

- If you encounter errors, ensure that all dependencies are properly installed.
- If you updated any data files, you may need to re-run `preprocess_train.py` and `preprocess_test.py` to ensure consistency.

For further details or troubleshooting, please refer to the project documentation or contact the project maintainer.

