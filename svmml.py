## backend/app.py
from flask import Flask, jsonify, request
from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib

# declare constants
HOST = '0.0.0.0'
PORT = 8081
# initialize flask application
app = Flask(__name__)


@app.route('/train', methods=['POST'])
def train():
    # get parameters from request
    parameters = request.get_json()

    # read iris data set
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # fit model
    clf = svm.SVC(C=float(parameters['C']),
                  probability=True,
                  random_state=1)
    clf.fit(X, y)
    # persist model
    joblib.dump(clf, 'E:/model.pkl')
    return jsonify({'accuracy': round(clf.score(X, y) * 100, 2)})


if __name__ == '__main__':
    # run web server
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT)
