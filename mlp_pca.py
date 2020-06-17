# PCA using a multilebel perceptron to use on a non0determined feature dataset
import numpy as np                                     # needed for arrays
import pandas as pd                                    # data frame
import matplotlib.pyplot as plt                        # modifying plot
from sklearn.model_selection import train_test_split   # splitting data
from sklearn.preprocessing import StandardScaler       # scaling data
from sklearn.decomposition import PCA                  # PCA package
from sklearn.metrics import accuracy_score             # grading
from sklearn.neural_network import MLPClassifier       # Multi-level perceptron
from sklearn.metrics import confusion_matrix            # Confusion Matrix
from warnings import filterwarnings                     # To ignore any convergence warnings
filterwarnings('ignore')

# Implementing the dataset and separating the values into x and y sets
data = pd.read_csv('sonar_all_data_2.csv',header=None)
X = data.iloc[:,:-2]
y = data.iloc[:,-2]

print(data)
# print(X)
# print(y)

# now split the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                                                    random_state=0)
stdsc = StandardScaler()            # apply standardization
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


componentsNumber = len(X.columns)
print(componentsNumber)
components = np.arange(1,componentsNumber + 1)        # Array for the Number of principle components
accuracy = np.zeros(len(components))        # Blank array for accuracy
max_accuracy = ()               #
print("Components \t Accuracy")

for index in components:
    pca = PCA(n_components = index)                 # Varying Components with for loop
    X_train_pca = pca.fit_transform(X_train_std)  # apply to the train data
    X_test_pca = pca.transform(X_test_std)  # do the same to the test data

    # Training to the model
    model = MLPClassifier(hidden_layer_sizes = (100,), activation='logistic',
                            max_iter = 2000, alpha = 0.00001, solver ='adam', tol = 0.0001)
    model.fit(X_train_pca, y_train)


    actual = y_test                             # Declaring the actural
    predicted = model.predict(X_test_pca)       # Declaring the predicted by the model

    accuracy[index-1] = accuracy_score(y_test, predicted)       # Testing the accuracy on the test model

    print(index, "\t\t\t", accuracy[index-1])           # Printing the Number of components with the corresponding accuracy

    # If loops that are constructed to save the max accuracy with the coreesponding actual and predicted
    # values for the confusion matrix
    if (index == 1):
        max_accuracy = accuracy[index-1]
    if (accuracy[index-1] > max_accuracy and index > 1):
        cmat_actual = actual
        cmat_predicted = predicted
        max_accuracy = accuracy[index - 1]
        max_index = index

# Nice Print statements and separators for the Max Accuracy and Component
print('-' * 40)
print("Maximum Accuracy is", max_accuracy)
print("Principle Components for Max Accuracy is", max_index)
print('-' * 40)

# Plotting of the accuracy vs components
plt.plot(components, accuracy)
plt.xlabel("Principle Components")
plt.ylabel("Accuracy")
plt.show()

# Printing the confusion matrix
print("Confusion Matrix")
cmat = confusion_matrix(cmat_actual, cmat_predicted)
print(cmat)