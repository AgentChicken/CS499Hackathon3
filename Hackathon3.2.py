from flask import Flask
from flask import render_template
from sklearn import linear_model
from flask_wtf import FlaskForm
from sklearn.preprocessing import PolynomialFeatures
from wtforms import SubmitField, IntegerField
from wtforms.validators import DataRequired
import sys


class SubmitData(FlaskForm):
    route = IntegerField('sroute', validators=[DataRequired()])
    direction = IntegerField('sdirection', validators=[DataRequired()])
    day = IntegerField('sday', validators=[DataRequired()])
    time = IntegerField('stime', validators=[DataRequired()])
    traffic = IntegerField('straffic', validators=[DataRequired()])
    submit = SubmitField('Submit')


class PredictData(FlaskForm):
    p_route = IntegerField('proute', validators=[DataRequired()])
    p_direction = IntegerField('pdirection', validators=[DataRequired()])
    p_day = IntegerField('pday', validators=[DataRequired()])
    p_time = IntegerField('ptime', validators=[DataRequired()])
    p_submit = SubmitField('Predict')

app = Flask(__name__)

X = [[-sys.maxint, -sys.maxint, -sys.maxint, -sys.maxint]]
vector = [-1]
poly = PolynomialFeatures(degree=6)
clf = linear_model.LinearRegression()
clf.fit(poly.fit_transform(X), vector)


def commit(route, direction, day, time, traffic):
    X.append([route, direction, day, time])
    vector.append(traffic)
    clf.fit(poly.fit_transform(X), vector)


def conjecture(route, direction, day, time):
    return clf.predict(poly.fit_transform([route, direction, day, time]))[0]


def process_input(element):
    return int(str((str(element).split("value=\""))[1].split("\">")[0]))


@app.route('/', methods=["GET", "POST"])
def login():
    form = SubmitData(csrf_enabled=False)
    predict = PredictData(csrf_enabled=False)
    if form.validate_on_submit():
        commit(process_input(form.route),
               process_input(form.direction),
               process_input(form.day),
               process_input(form.time),
               process_input(form.traffic))
    if predict.validate_on_submit():
        predicted = conjecture(process_input(predict.p_route),
                          process_input(predict.p_direction),
                          process_input(predict.p_day),
                          process_input(predict.p_time))
        if predicted == -1:
            result = "Not enough data. Please input data above."
        else:
            result = "Prediction: " + str(predicted)
        return render_template('index.html',
                               title='TRAFFIC PREDICTIONS',
                               form=form,
                               predict=predict,
                               result=result,
                               currentx=X[1:],
                               currentv=vector[1:])
    return render_template('index.html',
                           title='TRAFFIC PREDICTIONS',
                           form=form,
                           predict=predict,
                           result="",
                           currentx=X[1:],
                           currentv=vector[1:])


if __name__ == '__main__':
    app.run(host='0.0.0.0')
