CATGUARD - Predictive Maintenance

AI-powered predictive maintenance system for estimating industrial equipment failure risk using machine sensor data.

PROJECT OVERVIEW

CATGUARD is a student predictive-maintenance prototype that analyzes machine operating conditions and estimates the probability of equipment failure.

The system takes parameters such as machine type, air temperature, process temperature, rotational speed, torque, and tool wear as input and provides a machine-failure prediction through an interactive Streamlit dashboard.


PROBLEM STATEMENT

Unexpected equipment failures can cause:

- Machine downtime
- Increased maintenance costs
- Reduced operational efficiency
- Unexpected production interruptions

CATGUARD explores how historical machine sensor data can be used to identify potential failure conditions and support proactive maintenance decisions.


KEY FEATURES

- Machine failure probability prediction
- Interactive Streamlit dashboard
- Machine type selection
- Temperature monitoring
- Rotational speed analysis
- Torque monitoring
- Tool-wear analysis
- Failure-risk visualization
- Machine-learning based prediction


HOW CATGUARD WORKS

Machine Sensor Data
        |
        v
Data Preparation
        |
        v
Feature Processing
        |
        v
Machine Learning Model
        |
        v
Failure Probability
        |
        v
Streamlit Dashboard

The system follows these steps:

1. Historical machine sensor data is loaded.
2. The data is prepared for machine learning.
3. Machine type is converted into numerical features.
4. Training and testing datasets are created.
5. A logistic regression model is trained.
6. Class weighting is used because machine failures are relatively rare.
7. The trained model estimates failure probability.
8. The prediction is displayed through the Streamlit dashboard.


DATASET

The project uses the AI4I 2020 Predictive Maintenance Dataset.

The dataset contains machine operating measurements and recorded machine-failure information.

Input Features

Air Temperature - Temperature of the surrounding environment

Process Temperature - Temperature associated with the machine process

Rotational Speed - Speed of the rotating component

Torque - Rotational turning force

Tool Wear - Accumulated tool usage/wear

Machine Type - Machine category


MACHINE LEARNING MODEL

CATGUARD uses Logistic Regression for binary classification.

Logistic regression is a machine-learning method that estimates the probability of an event occurring. In this project, the event is machine failure.

The model analyzes the machine's operating parameters and produces a failure probability.


HANDLING RARE FAILURE CASES

Machine failures are much less common than normal operating conditions in the dataset.

To address this imbalance, class weighting is used during model training. This gives greater importance to failure examples and helps the model detect more actual failure cases.


MODEL PERFORMANCE

The model was evaluated on 2,000 test records.

Accuracy: 82.00%

Precision: 13.68%

Recall: 80.88%

F1 Score: 23.40%

The model detected 55 out of 68 actual failure cases in the test dataset.

The relatively low precision indicates that the current prototype generates a number of false alarms. This trade-off is documented as part of the current model's performance rather than being hidden.


DASHBOARD

CATGUARD provides an interactive dashboard where users can enter machine operating conditions and receive a failure probability.

Dashboard Results

Result1.png

Result2.png

Test Cases

Testcase1.png

Testcase2.png


TECHNOLOGIES USED

- Python
- Pandas
- NumPy
- Streamlit
- Git
- GitHub


PROJECT STRUCTURE

CATGUARD_Predictive_Maintanance/

    app.py
    README.md
    .gitignore

    data/
        ai4i2020.csv
        X_train.csv
        X_test.csv
        y_train.csv
        y_test.csv
        catguard_model.npz

    src/
        download_data.py
        explore_data.py
        predict.py
        prepare_data.py
        train_model.py

    screenshots/
        Result1.png
        Result2.png
        Testcase1.png
        Testcase2.png


HOW TO RUN

1. Clone the Repository

git clone https://github.com/brindhajs/CATGUARD_Predictive_Maintanance.git

2. Open the Project

cd CATGUARD_Predictive_Maintanance

3. Create a Virtual Environment

python -m venv .venv

4. Activate the Environment

For Windows PowerShell:

.venv\Scripts\Activate.ps1

5. Install Required Packages

pip install pandas numpy streamlit

6. Run the Dashboard

streamlit run app.py

The CATGUARD dashboard will open in your browser.


FUTURE IMPROVEMENTS

- Improved failure classification
- Additional machine-learning models
- Better handling of false alarms
- More detailed equipment-health visualization
- Historical prediction tracking
- Real-time sensor-data integration
- Maintenance recommendation features


DISCLAIMER

CATGUARD is a student predictive-maintenance prototype inspired by industrial equipment monitoring use cases.

It is not an actual Caterpillar system and is not intended for real-world production maintenance decisions.


AUTHOR

Brindha J.S.

B.Tech - Artificial Intelligence and Data Science
