import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('train.csv')
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
data = data.fillna(0)

dataset_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
#dataset_X = dataset_X.as_matrix()
x_train = dataset_X.as_matrix()

data['Deceased'] = data['Survived'].apply(lambda s: int(not s))
dataset_Y = data[['Deceased', 'Survived']]
#dataset_Y = dataset_Y.as_matrix()
y_train = dataset_Y.as_matrix()

#x_train, x_test, y_train, y_test = train_test_split(dataset_X, dataset_Y, test_size = 0.2, random_state = 42)

x = tf.placeholder(tf.float32, shape = [None, 6])
y = tf.placeholder(tf.float32, shape = [None, 2])

w = tf.Variable(tf.random_normal([6, 2]), name = 'weights')
b = tf.Variable(tf.zeros([2]), name = 'bias')
saver = tf.train.Saver()

y_pred = tf.nn.softmax(tf.matmul(x, w) + b)

cross_entropy = - tf.reduce_sum(y * tf.log(y_pred + 1e-10), reduction_indices = 1)
cost = tf.reduce_mean(cross_entropy)

train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	for epoch in range(100):
		total_loss = 0.
		for i in range(len(x_train)):
			feed = {x: [x_train[i]], y: [y_train[i]]}
			_, loss = sess.run([train_op, cost], feed_dict = feed)
			total_loss += loss
		print('Epoch: %04d, total loss = %.9f' % (epoch + 1, total_loss))
	print ('Training complete!')

#	pred = sess.run(y_pred, feed_dict = {x: x_test})
#	correct = np.equal(np.argmax(pred, 1), np.argmax(y_test, 1))
#	accuracy = np.mean(correct.astype(np.float32))
#	print("Accuracy on validation set: %.9f" % accuracy)

	save_path = saver.save(sess, "model/model.ckpt")

testdata = pd.read_csv('test.csv')
testdata = testdata.fillna(0)
testdata['Sex'] = testdata['Sex'].apply(lambda s: 1 if s == 'male' else 0)
x_test = testdata[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]

with tf.Session() as sess:
	saver.restore(sess, './model/model.ckpt')
	predictions = np.argmax(sess.run(y_pred, feed_dict = {x: x_test}), 1)

submission = pd.DataFrame({"PassengerId": testdata["PassengerId"], "Survived": predictions})
submission.to_csv("titanic-submission.csv", index = False)
