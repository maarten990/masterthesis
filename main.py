import sys
from data import Data, metadata_featurizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score


def main():
    data = Data(sys.argv[1], pattern='1800*.xml')
    X, y = data.sliding_window(3, metadata_featurizer)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f'{X_train.shape[0]} training samples, {X_test.shape[0]} testing samples')

    svm = LinearSVC()
    svm.fit(X_train, y_train)
    print('Accuracy: ', accuracy_score(y_test, svm.predict(X_test)))
    print('Speech recall: ', recall_score(y_test, svm.predict(X_test)))


if __name__ == '__main__':
    main()