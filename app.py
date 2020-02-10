from flask import Flask, render_template, request
from Functions import *
from keras import models
from keras.layers import Dense
import tensorflow as tf
global graph,model

graph = tf.get_default_graph()


model2 = models.Sequential()
model2.add(Dense(25, activation='relu', input_shape=(15000, )))
model2.add(Dense(16, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model2.load_weights('best_model.h5')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def submit():
    if request.method == 'POST':

        raw_review = request.form['review']
        review = tss(raw_review)
        review = vectorize_sentence(review)
        with graph.as_default():
            y = model2.predict(review)
        y = y[0][0]*100
        y = np.int(np.round(np.interp(y, [0.0, 100.0], [0,10])))

        return render_template("index.html", results=y)

if __name__ == "__main__":
    app.run(debug=True)