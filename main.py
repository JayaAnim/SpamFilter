from svm import SVM
import sys

class User:
    def __init__(self):
        #specifies method user wants to extract features with
        self.extract = None
        #specifies method user wants to select features with
        self.select = None
        #specifies variance threshold
        self.threshold = None
        #specifies k best features param
        self.k = None
        #specifies number of PCA components
        self.n = None

        self.getInput()

    def getInput(self):
        # Prompt the user for feature extraction method
        print('Please choose a feature extraction method:')
        print('1. Bag of Words')
        print('2. TF-IDF')
        while True:
            try:
                choice = int(input('Enter your choice (1 or 2): '))
                if choice in [1, 2]:
                    self.extract = choice
                    break
                else:
                    print('Invalid choice. Please enter 1 or 2.')
            except ValueError:
                print('Invalid input. Please enter an integer.')

        # Prompt the user for feature selection method
        print('\nPlease choose a feature selection method:')
        print('1. Remove low-variance features')
        print('2. Select k best features')
        print('3. Perform PCA')
        print('4. No feature selection')
        while True:
            try:
                choice = int(input('Enter your choice (1, 2, 3, or 4): '))
                if choice in [1, 2, 3, 4]:
                    self.select = choice
                    break
                else:
                    print('Invalid choice. Please enter 1, 2, 3, or 4.')
            except ValueError:
                print('Invalid input. Please enter an integer.')

        # If feature selection method requires additional parameters, prompt the user for them
        if self.select == 1:
            while True:
                try:
                    threshold = float(input('\nEnter the variance threshold (a float between 0 and 1, .4 is recommended): '))
                    if 0 < threshold < 1:
                        self.threshold = threshold
                        break
                    else:
                        print('Invalid threshold. Please enter a float between 0 and 1.')
                except ValueError:
                    print('Invalid input. Please enter a float.')

        elif self.select == 2:
            while True:
                try:
                    k = int(input('\nEnter the number of top features to select (an integer), 50 is recommended: '))
                    if k > 0:
                        self.k = k
                        break
                    else:
                        print('Invalid value. Please enter an integer greater than 0.')
                except ValueError:
                    print('Invalid input. Please enter an integer.')

        elif self.select == 3:
            while True:
                try:
                    n = int(input('\nEnter the number of principal components to keep (an integer), 500 is recommended: '))
                    if n > 0:
                        self.n = n
                        break
                    else:
                        print('Invalid value. Please enter an integer greater than 0.')
                except ValueError:
                    print('Invalid input. Please enter an integer.')




def welcomeUser():
    print('/* --------- Disclaimer ---------- */')
    print('This program is resource intensive and some methods can take 10-15 methods.')
    print('If your computer is low-spec, selecting a feature selection method is recommended.')
    print('/* -------------------------------- */ \n')
    print('================ Welcome =================\n')


def runProgramAgain():
    while True:
        run_again = input("Do you want to run the program again? (yes/no): ")
        if run_again.lower() == "no":
            sys.exit()
        elif run_again.lower() == "yes":
            print("Running program again...\n")
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

def main():
    welcomeUser()
    while True:
        user = User()
        svm = SVM(user)
        runProgramAgain()

main()
        
