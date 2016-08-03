import tensorflow as tf
from numpy import asarray, reshape
from sklearn.cross_validation import train_test_split
import pandas as pd


def one_hot_encode(response):
  """ Creates a one-hot encoding of
  of response variables.
  Require response dict which is output dict f
  rom read_mapfile() function
  """
  ohe_dict = {}
  resp_categ = list(set(response.values()))
  for x in resp_categ:
    ohe_store = [0]*len(resp_categ)
    ohe_store[resp_categ.index(x)] = 1
    ohe_dict[x] = ohe_store
  return ohe_dict

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride_feat):
  return tf.nn.conv2d(x, W, strides=[1, 1, stride_feat, 1], padding='SAME')

# ksize: A list of ints that has length >= 4. 
# ksize is the size of the window for each dimension of the input tensor.
# stride: A list of ints that has length >= 4. 
# stride is the stride of the sliding window for each dimension of the input tensor.
# 3rd element in the list of ksize indicates the number of features/OTUs in dataset
# Changing ksize feature value to 2 doesn't impact much. 
# Changing to 15 also has little impact.

def max_pool_n(x,maxp_k,maxp_str):
  return tf.nn.max_pool(x, ksize=[1, 1, maxp_k, 1],
                        strides=[1, 1, maxp_str, 1], padding='SAME')


def conv_layer(input_feat,conv_feature,x_shape,conv2d_stride,
  maxpool_ksize,maxpool_stride,in_ch):
  """ Convolutional layer """
  W_conv1 = weight_variable([1, input_feat, in_ch, conv_feature])
  b_conv1 = bias_variable([conv_feature])
  h_conv1 = tf.nn.relu(conv2d(x_shape, W_conv1,conv2d_stride) + b_conv1)
  #red_feat_dim1 = h_conv1.get_shape().as_list()[2]
  h_pool1 = max_pool_n(h_conv1,maxpool_ksize,maxpool_stride)
  # In order to get the reduced dimension of feature vector, we do following:
  # 2 is the index of the 4d tensor
  red_feat_dim1 = h_pool1.get_shape().as_list()[2]
  if (red_feat_dim1 < conv2d_stride):
    conv2d_stride = red_feat_dim1
  if (red_feat_dim1 < maxpool_stride):
    maxpool_stride = red_feat_dim1
  if (red_feat_dim1 < maxpool_ksize):
    maxpool_ksize = red_feat_dim1
  return [h_pool1,red_feat_dim1,conv2d_stride,maxpool_ksize,maxpool_stride,conv_feature]


#### Variables to read
otuinfile = 'dgerver_cd_otu.txt'
metadata = 'mapfile_dgerver.txt'
# Column name from mapfile for disease
col = 'gastrointest_disord'
# Split 55% of data as training and 45% as test
train_ratio = 0.55


# Function calls
# 335 OTUs in total and 490 samples

#### Variables to read
otuinfile = 'glne007.final.an.unique_list.0.03.subsample.0.03.filter.shared'
mapfile = 'metadata.tsv'
# Column name from mapfile for disease
col = 'dx'
# Split 55% of data as training and 45% as test
train_ratio = 0.45

# OTU data
data = pd.read_table(otuinfile,sep='\t',index_col=1)
filtered_data = data.dropna(axis='columns', how='all')
filtered_data = filtered_data.drop(['label','numOtus'],axis=1)
# Meta-data
metadata = pd.read_table(mapfile,sep='\t',index_col=0)
dx = metadata[col]

response = {}
for x in range(dx.shape[0]):
  response[list(dx.index)[x]] = list(dx.ravel())[x]
ohe_dict = one_hot_encode(response)

X, P, Y, Q = train_test_split(
filtered_data, dx, test_size=train_ratio, random_state=42)
train_dataset = pd.concat([X, Y], axis=1)
test_dataset = pd.concat([P, Q], axis=1)

## Create test data for TF
test_input = []
test_output = []
for index, row in test_dataset.iterrows():
  otudat = row.drop(col, axis=0).values
  oresp = row[col]
  test_input.append(otudat)
  test_output.append(ohe_dict[oresp])


### Extra check for batch
#train_input = []
#train_output = []
#for index, row in train_dataset.iterrows():
#  otudat = row.drop(col, axis=0).values
#  oresp = row[col]
#  train_input.append(otudat)
#  train_output.append(ohe_dict[oresp])




# Data specific:
# Get features and levels of response variable
feature = train_dataset.shape[1]-1
resp = len(ohe_dict)


# These variables are not having impact on accuracy
first_conv_feature = 32
second_conv_feature = 64
dense_layer_feature = 1024
opt_param = 1e-1
# For AdadeltaOptimizer()
a = 0.1  # learning_rate
b = 0.95  # rho
c = 1e-02  # epsilon


# Value to stride for conv2d
conv2d_stride = 1
# Value to stride and ksize for maxpool
maxpool_ksize = 1
maxpool_stride = 1
str_siz = [conv2d_stride,maxpool_ksize,maxpool_stride]

x = tf.placeholder(tf.float32, [None, feature])
y_ = tf.placeholder(tf.float32, [None, resp])
# Reshape 267 features into 1*267 matrix
x_shape = tf.reshape(x, [-1,1,feature,1])


# Call convolutional layers
# The last argument, 1, is the value of first convolution input channel
conv1 = conv_layer(feature,first_conv_feature,x_shape,
  str_siz[0],str_siz[1],str_siz[2],1)
conv2 = conv_layer(conv1[1],second_conv_feature,conv1[0],
  conv1[2],conv1[3],conv1[4],conv1[5])


# Densely Connected Layer
# Values of 23 * 1 * 64 come from printing (h_pool2)
# If ksize is 2, then feature value is 67 for densly connected layer
W_fc1 = weight_variable([1 * conv2[1] * second_conv_feature, dense_layer_feature])
b_fc1 = bias_variable([dense_layer_feature])
# Not doing pooling
h_pool2_flat = tf.reshape(conv2[0], [-1, 1*conv2[1]*second_conv_feature]) #h_conv2
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
# Dropout or no drop-out has no impact
k_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, k_prob)

# Readout Layer
W_fc2 = weight_variable([dense_layer_feature, resp])
b_fc2 = bias_variable([resp])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) #h_fc1_drop

# Train and evaluate
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))


#AdamOptimizer(opt_param)
train_step = tf.train.AdadeltaOptimizer(a,b,c).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.Session()
sess.run(tf.initialize_all_variables())

for index, row in train_dataset.iterrows():
  # Store 0th-index is HIV postive and 1st-index is HIV negative
  otudat = row.drop(col, axis=0).values
  response = ohe_dict[row[col]]
  batch_xs = reshape(otudat, (-1, feature))
  batch_ys = reshape(response, (-1, resp))
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, k_prob: 0.5})

#sess.run(train_step, feed_dict={x: asarray(train_input), y_: asarray(train_output), k_prob: 0.5})
print(sess.run(accuracy, feed_dict={x: asarray(test_input), y_: asarray(test_output),k_prob: 1.0}))