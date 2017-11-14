import flask
import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier


#------- CONFIG --------#
app = flask.Flask(__name__) # initialise Flask app var
app.config['DEBUG'] = True

#-------- MODEL -----------#

df = pd.read_csv('titanic.csv')
include = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Survived']

# Create dummies and drop NaNs
df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'male' else 1)
df = df[include].dropna()

X = df[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp']]
y = df['Survived']

PREDICTOR = RandomForestClassifier(n_estimators=100).fit(X, y)


#with open('picked_model.pkl', 'w') as picklefile:
#    pickle.dump(picklefile, PREDICTOR)


#with open('picked_model.pkl', 'r') as picklefile:
#    PREDICTOR = pickle.load(picklefile)


#------- ROUTES --------#
@app.route("/")
def hello():
    #return "Hello World!"
    return '''
    <body>
    <h2> Hello World! <h2>
    </body>
    '''

@app.route('/greet/<name>')
def greet(name):
    '''Say hello to your first parameter'''
    return "Hello, %s!" %name
    
@app.route('/input_page')
def send_form():
  return flask.render_template('input_page.html')
  ## flask.redirect
  ## return full html
  
  ## flask using session over local storage or web sql
  ## user multifiable
  
  
@app.route('/make_it_happen', methods=['POST'])

## REST API -- POST,GET,PUT,DELETE
def say_hi():
  name = flask.request.form['myname']
  excitement_level = flask.request.form['mylevel']
  return flask.render_template('stuff_you_know.html', name=name, lvl=excitement_level)
  
@app.route('/predict', methods=["GET"])
def predict():
    pclass = flask.request.args['pclass']
    sex = flask.request.args['sex']
    age = flask.request.args['age']
    fare = flask.request.args['fare']
    sibsp = flask.request.args['sibsp']

    item = np.array([pclass, sex, age, fare, sibsp]).reshape(1,-1)
    score = PREDICTOR.predict_proba(item)
    results = {'survival chances': score[0,1], 'death chances': score[0,0]}
    return flask.jsonify(results)
 

#----- ROUTES -----#

# This method takes input via an HTML page
@app.route('/predict2')
def predict2():
   return flask.render_template("titanic_input.html")

@app.route('/result', methods=['POST'])
def result():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':

       inputs = flask.request.form

       pclass = inputs['pclass'][0]
       sex = inputs['sex'][0]
       age = inputs['age'][0]
       fare = inputs['fare'][0]
       sibsp = inputs['sibsp'][0]

       item = np.array([pclass, sex, age, fare, sibsp]).reshape(1, -1)
       score = PREDICTOR.predict_proba(item)
       results = {'survival chances': score[0,1], 'death chances': score[0,0]}
       return flask.jsonify(results) 
 
 
#-------- MAIN  SENTINEL----------#
if __name__ == '__main__':
    app.run()