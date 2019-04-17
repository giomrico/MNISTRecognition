from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from random import randint
from sklearn.linear_model import LogisticRegression


def random_number(n):
    start = 10**(n-1)
    end = (10**n)-1
    return randint(start, end)

#Took this function from our jupter PA4 lab, useful to show certain connctions between prdicted and true
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="center", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #print(cm.shape[0],cm.shape[1], '------------------------------------')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


mnist = fetch_mldata('MNIST original')
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=.3, random_state=42)


randomDigit = random_number(3)
plt.figure(figsize=(20, 6))


'''
This code is used to give us an insight on what
both the images them selves look like as well as
what the labels look like.

As we can see, the images are the numbers in the graph
and the labels are the numbers
'''


count = randomDigit-5


while count < randomDigit:
    numberImage = X_train[count]
    actualNumber = y_train[count]
    plt.imshow(np.reshape(numberImage, (28, 28)))
    plt.title('Training Number: %i\n' % actualNumber, fontsize=15)
    plt.axis('off')
    plt.show()
    count += 1




logisticRegr = LogisticRegression(solver='lbfgs', max_iter=100)
logisticRegr.fit(X_train, y_train)
predictions = logisticRegr.predict(X_test)


index = 0
wrongIndex = []


'''
This code is used to get the indexes of these images
whose predictions were not concurrent with the True value
of the label
'''
count = 0
while count < len(y_test):
    actualNumber = y_test[count]
    prediction = predictions[count]
    if actualNumber != prediction:
        wrongIndex.append(index)
        index += 1
    count += 1


plt.figure(figsize=(20, 6))


'''
This code is used to show a mix of data from both wrong
and right predictions, it was done this was to show how
the predictions even when wrong can be seen as reasonable
and also to showcase its actual power on real numbers
'''
count = randomDigit-5
while count < randomDigit:
    index = wrongIndex[count]
    plt.imshow(np.reshape(X_test[index], (28,28)))
    plt.title('Prediction = {}, Correct = {}'.format(predictions[index], y_test[index]), fontsize=15)
    plt.axis('off')
    plt.show()
    count += 1


target_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
score = logisticRegr.score(X_test, y_test)
plot_confusion_matrix(y_test, predictions, classes=target_labels, normalize=True, title="Test Accuracy: %.2f %%"% (score*100))
plt.show()
