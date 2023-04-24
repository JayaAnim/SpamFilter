# Feature selection and extraction comparison

# Introduction

main.py creates a SVM object which allows the user to specify the feature extraction and selection methods used. Once selected the program will train the SVM appropriately and display the accuracy results. The SVM is built to detect spam messages and is tested on on the spam.csv dataset. 

It allows users to compare methods of filtering spam and determine which is the best fit for their use case.

# Installation

To install the necessary dependencies for this project, you can use pip with the included requirements.txt file:

First, ensure that you have Python 3 installed on your system. If you do not have Python 3 installed, you can download it from the official website: https://www.python.org/downloads/

Next, create a new virtual environment using the venv module that comes with Python 3:
    python3 -m venv env

Activate the virtual environment:
    On macOS and Linux:
        source env/bin/activate
    On Windows:
        .\env\Scripts\activate

Install the dependencies using the following command:
    pip install -r requirements.txt

This will install all the required packages specified in the requirements.txt file.

Note: If you are using a different package manager like Anaconda or Miniconda, the commands may be slightly different.

# Usage

To start the program:
    python3 main.py

1) The program will ask the user which feature extraction method to use (BoW or TF-IDF)

2) The program will ask the user which feature selection method to use (variance selection, chi squared test, pca, or none)

3) The program will ask the user for additional needed input for feature selection

4) The program will begin training the SVM appropriately, displaying what step is currently being run due to high load times

5) The program will display the accuracy results and ask the user if he/she would like to rerun the program

# Example

![Example run](https://github.com/JayaAnim/SpamFilter/blob/main/example.png)