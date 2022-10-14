# Necessary libraries
from scipy.linalg import fractional_matrix_power
import tensorflow as tf
import pandas as pd
import scipy.sparse as sp
import numpy as np
import math
from scipy.ndimage.interpolation import shift

tf.compat.v1.enable_eager_execution()
import warnings
warnings.filterwarnings("ignore")

from config import DEFINES
from tensorflow.keras.backend import eval

def Model_DRLIR(features, labels, mode, params):

    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT



    positional_encoded = positional_encoding(params['embedding_size'], params['max_sequence_length'])
    positional_encoded.trainable = True
    if TRAIN:
        position_inputs = tf.tile(tf.range(0, params['max_sequence_length']), [params['batch_size']])
        position_inputs = tf.reshape(position_inputs, [params['batch_size'], params['max_sequence_length']])
    else:
        position_inputs = tf.tile(tf.range(0, params['max_sequence_length']), [1])
        position_inputs = tf.reshape(position_inputs, [1, params['max_sequence_length']])

    embedding = tf.compat.v1.get_variable(name='embedding', shape=[params['vocabulary_length'], params['embedding_size']], initializer=tf.keras.initializers.GlorotUniform())
    W_e = tf.tile(tf.expand_dims(tf.compat.v1.get_variable("W_e", [params['embedding_size'], params['embedding_size']], dtype=tf.float32), 0), [params['batch_size'], 1, 1])
    W_p = tf.tile(tf.expand_dims(tf.compat.v1.get_variable("W_p", [params['embedding_size'], params['embedding_size']], dtype=tf.float32), 0),[params['batch_size'], 1, 1])

    W_d = tf.tile(tf.expand_dims(tf.compat.v1.get_variable("W_d", [params['embedding_size'], params['embedding_size']], dtype=tf.float32), 0), [params['batch_size'], 1, 1])

    W_r = tf.compat.v1.tile(tf.expand_dims(tf.compat.v1.get_variable('W_r', [1, params['embedding_size']], dtype=tf.float32), 0),[params['batch_size'], 1, 1])
    W_co = tf.compat.v1.tile(tf.expand_dims(tf.compat.v1.get_variable('W_co', [1, params['embedding_size']], dtype=tf.float32), 0),[params['batch_size'], 1, 1])

    embedding_user = tf.compat.v1.get_variable(name='embedding_user', shape=[params['user_length'], params['embedding_size']], initializer=tf.keras.initializers.GlorotUniform())
    W_u= tf.tile(tf.expand_dims(tf.compat.v1.get_variable('W_u', [params['embedding_size'], params['embedding_size']], dtype=tf.float32), 0) , [params['batch_size'], 1, 1])

    embedding_week = tf.compat.v1.get_variable(name='embedding_week', shape=[8, params['embedding_size']], initializer=tf.keras.initializers.GlorotUniform())
    W_week = tf.tile(tf.expand_dims(tf.compat.v1.get_variable('W_week', [params['embedding_size'], params['embedding_size']], dtype=tf.float32), 0),[params['batch_size'], 1, 1])

    embedding_month = tf.compat.v1.get_variable(name='embedding_month', shape=[33, params['embedding_size']], initializer=tf.keras.initializers.GlorotUniform())
    W_month = tf.tile(tf.expand_dims(tf.compat.v1.get_variable('W_month', [params['embedding_size'], params['embedding_size']], dtype=tf.float32), 0), [params['batch_size'], 1, 1])

    embedding_day = tf.compat.v1.get_variable(name='embedding_day', shape=[25, params['embedding_size']], initializer=tf.keras.initializers.GlorotUniform())
    W_day = tf.tile(tf.expand_dims(tf.compat.v1.get_variable('W_day', [params['embedding_size'], params['embedding_size']], dtype=tf.float32), 0), [params['batch_size'], 1, 1])

    embedding_season = tf.compat.v1.get_variable(name='embedding_season', shape=[5, params['embedding_size']], initializer=tf.keras.initializers.GlorotUniform())
    W_season = tf.tile(tf.expand_dims(tf.compat.v1.get_variable('W_season', [params['embedding_size'], params['embedding_size']], dtype=tf.float32), 0), [params['batch_size'], 1, 1])
    u_hot = tf.nn.embedding_lookup(embedding_user, tf.cast(features['in_users'], tf.int32))
    encoder_user_features = tf.matmul(u_hot, W_u)
    # print(" Encoder user features = ")
    # print(u_hot)

    recent_time_features = tf.matmul(tf.expand_dims(tf.cast(features['in_time'], tf.float32), 2), W_r)

    encoder_week_features = tf.matmul(tf.nn.embedding_lookup(embedding_week, tf.cast(features['in_week'], tf.int32)), W_week)
    encoder_month_features = tf.matmul(tf.nn.embedding_lookup(embedding_month, tf.cast(features['in_mon'], tf.int32)), W_month)
    encoder_season_features = tf.matmul(tf.nn.embedding_lookup(embedding_season, tf.cast(features['in_sea'], tf.int32)), W_season)  #
    encoder_day_features = tf.matmul(tf.nn.embedding_lookup(embedding_day, tf.cast(features['in_day'], tf.int32)), W_day)  #
    periodic_input_features = tf.divide((encoder_week_features + encoder_month_features + encoder_season_features + encoder_day_features +recent_time_features),5.0)

    encoder_inputs = tf.matmul(tf.nn.embedding_lookup(embedding, tf.cast(features['input'], tf.int32)), W_e)
    decoder_inputs = tf.matmul(tf.nn.embedding_lookup(embedding, tf.cast(features['output'], tf.int32)), W_d)


    GCN_values = params['gcn_values']
    encoder_inputs_GCN = tf.matmul(tf.nn.embedding_lookup(tf.cast(GCN_values, tf.float32), tf.cast(features['input'], tf.int32)), W_p)
    decoder_inputs_GCN = tf.matmul(tf.nn.embedding_lookup(tf.cast(GCN_values, tf.float32), tf.cast(features['output'], tf.int32)), W_p)

    position_inputs

    position_encode = tf.nn.embedding_lookup(positional_encoded, position_inputs)

    # encoder_inputs = tf.tensordot(positional_encoded, encoder_inputs, axes = 1)
    # encoder_inputs_GCN = tf.tensordot(positional_encoded, encoder_inputs_GCN)
    # periodic_input_features =  tf.tensordot(positional_encoded, periodic_input_features)
    # encoder_user_features = tf.tensordot(positional_encoded, encoder_user_features)

    # encoder_inputs_GCN = itemwiseproduce(encoder_inputs_GCN, position_encode)
    # periodic_input_features = itemwiseproduce(periodic_input_features, position_encode)
    # periodic_input_features = itemwiseproduce(periodic_input_features, position_encode)




    if (params['G_P_Q'] == 1):
        encoder_inputs = encoder_inputs + encoder_inputs_GCN  + periodic_input_features +  encoder_user_features
    elif(params['G_P_Q'] == 2):
        encoder_inputs =encoder_inputs+ periodic_input_features+encoder_user_features
    elif (params['G_P_Q'] ==3):
        encoder_inputs = encoder_inputs+ periodic_input_features+encoder_user_features
    elif (params['G_P_Q'] == 4):
        encoder_inputs = encoder_inputs+encoder_inputs_GCN+periodic_input_features
    elif (params['G_P_Q'] == 5):
        encoder_inputs = encoder_inputs+encoder_inputs_GCN+periodic_input_features+encoder_user_features

    encoder_inputs = encoder_inputs + position_encode
    decoder_inputs = decoder_inputs + position_encode
    # dmodel = 512, inner-layer has dimensionality df f = 2048.  (512 * 4)
    # dmodel = 128 , inner-layer has dimensionality df f = 512  (128 * 4)
    # H = 8 N = 6
    # H = 4 N = 2
    encoder_outputs = encoder(encoder_inputs,
                              [params['hidden_size'] * 4, params['hidden_size']], params['heads_size'], params['layers_size'])
    decoder_outputs = decoder(decoder_inputs,
                              encoder_outputs,
                              [params['hidden_size'] * 4, params['hidden_size']], params['heads_size'],params['layers_size'])
    # print(decoder_outputs.shape)
    # decoder_outputs = tf.reshape(decoder_outputs, [decoder_outputs.shape[0], -1])
    # print(decoder_outputs.shape)
    logits = tf.keras.layers.Dense(params['vocabulary_length'])(decoder_outputs)


    if PREDICT:

        decoder_outputs1 = tf.reshape(decoder_outputs, [decoder_outputs.shape[0], -1])
        logits1 = tf.keras.layers.Dense(params['vocabulary_length'])(decoder_outputs1)
        predictions = {
            'topk': tf.nn.top_k(logits1,  params['vocabulary_length'] )[1],#   tf.nn.top_k(logits1, params['vocabulary_length'])[1],          #
            # 'logit': logits,
            'reward': tf.nn.top_k(logits1,  params['vocabulary_length'] )[0] # tf.nn.top_k(logits1, params['vocabulary_length'])[0]   #
        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    # print("labels shape = ", labels.get_shape())
    decoder_outputs1 = tf.reshape(decoder_outputs, [decoder_outputs.shape[0], -1])
    logits1 = tf.keras.layers.Dense(params['vocabulary_length'])(decoder_outputs1)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=labels[:,0]))

    if EVAL:


        correct_prediction_5_mean = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.tile(labels[:,0:1],[1,5]), tf.int32), tf.nn.top_k(logits1[:,:],5)[1]), tf.float32)) #/ (tf.cast(DEFINES.batch_size, tf.float32)*5) #*5 #/ tf.cast((tf.shape(logits)[0] * tf.shape(logits)[1]),tf.float32) #DEFINES.batch_size * params['vocabulary_length'])
        correct_prediction_10_mean = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.tile(labels[:, 0:1], [1, 10]), tf.int32), tf.nn.top_k(logits1[:,:], 10)[1]),tf.float32))  #/ (tf.cast(DEFINES.batch_size, tf.float32)*5)  #*10

        precision_5 = (correct_prediction_5_mean, correct_prediction_5_mean)
        recall_5 = (correct_prediction_5_mean*5, correct_prediction_5_mean*5)

        precision_10 = (correct_prediction_10_mean, correct_prediction_10_mean)
        recall_10 = (correct_prediction_10_mean*10, correct_prediction_10_mean*10)

        f1_5 = (2*recall_5[0]*precision_5[0] /(recall_5[0] + precision_5[0] + 1e-8), 2*recall_5[0]*precision_5[0] /(recall_5[0] + precision_5[0] + 1e-8))
        f1_10 = (2 * recall_10[0]* precision_10[0] / (recall_10[0] + precision_10[0] + 1e-8), 2 * recall_10[0]* precision_10[0] / (recall_10[0] + precision_10[0] + 1e-8))


        idcg_5, idcg_10 = find_idcg()
        ndcg_5 = tf.reduce_mean(tf.math.log(2.0) / (tf.math.log(tf.cast(tf.where(tf.cast(tf.equal(tf.cast(tf.tile(labels[:, 0:1], [1, 1]), tf.int32), tf.nn.top_k(logits[:, 0:1, :], 5)[1]), tf.int64)),tf.float32) + 2.0)  ))  / idcg_5 #* tf.cast(DEFINES.batch_size, tf.float32))#
        ndcg_10 = tf.reduce_mean(tf.math.log(2.0) / (tf.math.log(tf.cast(tf.where(tf.cast(tf.equal(tf.cast(tf.tile(labels[:, 0:1], [1, 1]), tf.int32), tf.nn.top_k(logits[:, 0:1, :], 10)[1]), tf.int64)),tf.float32) + 2.0) ) )/ idcg_10 #* tf.cast(DEFINES.batch_size, tf.float32))

        ndcg_5 = (ndcg_5, ndcg_5)
        ndcg_10 = (ndcg_10, ndcg_10)

        metrics = {'precision_5': precision_5, 'recall_5': recall_5, 'f1_5': f1_5, 'ndcg_5': ndcg_5,'precision_10': precision_10, 'recall_10': recall_10, 'f1_10': f1_10, 'ndcg_10': ndcg_10}

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops= metrics)

    assert TRAIN


    # lrate = d−0.5 *  model · min(step_num−0.5, step_num · warmup_steps−1.5)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=DEFINES.learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    #
    # global_step = tf.Variable(0, trainable=False)
    # lr = tf.compat.v1.train.exponential_decay(DEFINES.learning_rate, global_step, DEFINES.train_steps, 0.1, staircase=True)
    #
    # # lrate = d−0.5 *  model · min(step_num−0.5, step_num · warmup_steps−1.5)
    # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr) #DEFINES.learning_rate)
    # train_op = optimizer.minimize(loss, global_step=global_step) #tf.compat.v1.train.get_global_step())
    #
    # return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def itemwiseproduce(a,b):
    c = tf.reshape(a,[-1])
    d = tf.reshape(b,[-1])
    res = tf.reshape(c*d,a.shape)
    return res

def positional_encoding(dim, sentence_length, dtype=tf.float32):
    #Positional Encoding
    # paper: https://arxiv.org/abs/1706.03762
    # P E(pos,2i) = sin(pos/100002i/dmodel)
    # P E(pos,2i+1) = cos(pos/100002i/dmodel)
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)



def encoder_module(inputs, num_units, heads):
    self_attn = sublayer_connection(inputs, multi_head_attention(inputs, inputs, inputs, heads))

    if DEFINES.conv_1d_layer:
        network_layer = conv_1d_layer(self_attn, num_units)
    else:
        network_layer = feed_forward(self_attn, num_units)

    outputs = sublayer_connection(self_attn, network_layer)
    return outputs


def decoder_module(inputs, encoder_outputs, num_units, heads):
    # sublayer_connection Parameter input Self-Attention
    # multi_head_attention parameter Query Key Value Head masked
    masked_self_attn = sublayer_connection(inputs, multi_head_attention(inputs, inputs, inputs, heads, masked=True))

    self_attn = sublayer_connection(masked_self_attn, multi_head_attention(masked_self_attn, encoder_outputs, encoder_outputs, heads))

    if DEFINES.conv_1d_layer:
        network_layer = conv_1d_layer(self_attn, num_units)
    else:
        network_layer = feed_forward(self_attn, num_units)

    outputs = sublayer_connection(self_attn, network_layer)
    return outputs



def encoder(inputs, num_units, heads, num_layers):
    outputs = inputs
    for _ in range(num_layers):
        outputs = encoder_module(outputs, num_units, heads)

    return outputs


def decoder(inputs, encoder_outputs, num_units, heads, num_layers):
    outputs = inputs
    for _ in range(num_layers):
        outputs = decoder_module(outputs, encoder_outputs, num_units, heads)

    return outputs


def layer_norm(inputs, eps=1e-6):
    # LayerNorm(x + Sublayer(x))
    feature_shape = inputs.get_shape()[-1:]
    # Pass the mean and standard deviation.
    mean = tf.keras.backend.mean(inputs, [-1], keepdims=True)
    std = tf.keras.backend.std(inputs, [-1], keepdims=True)
    beta = tf.Variable(tf.zeros(feature_shape), trainable=False)
    gamma = tf.Variable(tf.ones(feature_shape), trainable=False)

    return gamma * (inputs - mean) / (std + eps) + beta


def sublayer_connection(inputs, sublayer):
    # LayerNorm(x + Sublayer(x))
    return tf.keras.layers.Dropout(rate=0.5)(layer_norm(inputs + sublayer)) # Drop out rate 0.5


def feed_forward(inputs, num_units):
    # FFN(x) = max(0, xW1 + b1)W2 + b2
    with tf.compat.v1.variable_scope("feed_forward", reuse=tf.compat.v1.AUTO_REUSE):
        outputs = tf.keras.layers.Dense(num_units[0], activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dropout(rate=0.5)(outputs)  # Drop our rate 0.5
        return tf.keras.layers.Dense(num_units[1])(outputs)


def scaled_dot_product_attention(query, key, value, masked=False):
    #Attention(Q, K, V ) = softmax(QKt / root dk)V
    key_seq_length = float(key.get_shape().as_list()[-2])
    key = tf.transpose(key, perm=[0, 2, 1])
    outputs = tf.matmul(query, key) / tf.sqrt(key_seq_length)

    if masked:
        diag_vals = tf.ones_like(outputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        masks = tf.compat.v1.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

        paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

    attention_map = tf.nn.softmax(outputs)

    return tf.matmul(attention_map, value)


def multi_head_attention(query, key, value, heads, masked=False):
    # MultiHead(Q, K, V ) = Concat(head1, ..., headh)WO
    with tf.compat.v1.variable_scope("multi_head_attention", reuse=tf.compat.v1.AUTO_REUSE):
        feature_dim = query.get_shape().as_list()[-1]

        query = tf.keras.layers.Dense(feature_dim, activation=tf.nn.relu)(query)
        key = tf.keras.layers.Dense(feature_dim, activation=tf.nn.relu)(key)
        value = tf.keras.layers.Dense(feature_dim, activation=tf.nn.relu)(value)

        query = tf.concat(tf.split(query, heads, axis=-1), axis=0)
        key = tf.concat(tf.split(key, heads, axis=-1), axis=0)
        value = tf.concat(tf.split(value, heads, axis=-1), axis=0)

        attention_map = scaled_dot_product_attention(query, key, value, masked)

        attn_outputs = tf.concat(tf.split(attention_map, heads, axis=0), axis=-1)

        return attn_outputs


def conv_1d_layer(inputs, num_units):
    # Another way of describing this is as two convolutions with kernel size 1
    with tf.variable_scope("conv_1d_layer", reuse=tf.AUTO_REUSE):
        outputs = tf.keras.layers.Conv1D(num_units[0], kernel_size = 1, activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dropout(rate=DEFINES.dropout_width)(outputs)
        return tf.keras.layers.Conv1D(num_units[1], kernel_size = 1)(outputs)



def find_idcg():
    idcg_5, idcg_10 = 0.0, 0.0

    for i in range(5):
        idcg_5 = idcg_5 + tf.math.log(2.0) / tf.math.log(float(i) + 2)

    for i in range(10):
        idcg_10 = idcg_10 + tf.math.log(2.0) / tf.math.log(float(i) + 2)

    return idcg_5, idcg_10



def rearrange(pois, in_time,  in_users, day_seq, week_seq, mon_seq, sea_seq, output, out_t, out_users, target):

    features = {"input": pois,  "in_time":in_time,  "in_users":in_users, "in_day":day_seq, "in_week": week_seq, "in_mon":mon_seq,"in_sea":sea_seq,"output": output, "out_time":out_t,  "out_users": out_users}
    return features, target

# This is a function that goes into learning and creates batch data.
def train_input_fn(train_input, train_target_dec, batch_size):
    (train_poi_sequence_x, train_recent_time_sequence_x,train_user_sequence_x, train_periodic_sequence) = train_input
    train_poi_seqence_x_out = np.asarray( [np.concatenate(([1.0],y[0:DEFINES.max_sequence_length-1]),axis=None) for y in train_poi_sequence_x])

    train_recent_time_sequence_x_out = np.asarray([np.concatenate(([1.0],y[0:DEFINES.max_sequence_length-1]),axis=None) for y in train_recent_time_sequence_x])
    train_user_sequence_x_out = np.asarray([np.concatenate(([1.0],y[0:DEFINES.max_sequence_length-1]),axis=None) for y in train_user_sequence_x])

    (day_train_set_sequence, week_train_set_sequence, month_train_set_sequence, season_train_set_sequence) = train_periodic_sequence
    dataset = tf.data.Dataset.from_tensor_slices((train_poi_sequence_x, train_recent_time_sequence_x, train_user_sequence_x, day_train_set_sequence, week_train_set_sequence, month_train_set_sequence, season_train_set_sequence, train_poi_seqence_x_out, train_recent_time_sequence_x_out, train_user_sequence_x_out, train_target_dec))

    # Decay entire data
    dataset = dataset.shuffle(buffer_size=len(train_poi_sequence_x))
    # If there is no batch argument value, an error occurs.
    assert batch_size is not None, "train batchSize must not be None"
    # Sharing through from_tensor_slices
    # Bundle by batch size.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # For each element of data, use the rearrange function Convert elements through  # and compose them into maps.
    dataset = dataset.map(rearrange)
    # If you can put the desired number of epochs in the repeat () function,
    # If there are no arguments, iterators are infinite.
    dataset = dataset.repeat()

    # iterator through make_one_shot_iterator
    # Make it.
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset) #dataset.make_one_shot_iterator()
    # Tensor of next item via iterator
    # Give the object.

    return iterator.get_next()


## This is a function that goes into evaluation and creates batch data.
def eval_input_fn(eval_input, eval_target_dec, batch_size):
    (eval_poi_sequence_x,  eval_recent_time_sequence_x, eval_user_sequence_x, eval_periodic_sequence) =eval_input
    eval_poi_seqence_x_out = np.asarray([np.concatenate(([1.0], y[0:DEFINES.max_sequence_length - 1]), axis=None) for y in eval_poi_sequence_x])

    eval_recent_time_sequence_x_out = np.asarray([np.concatenate(([1.0], y[0:DEFINES.max_sequence_length - 1]), axis=None) for y in eval_recent_time_sequence_x])
    eval_user_sequence_x_out = np.asarray([np.concatenate(([1.0], y[0:DEFINES.max_sequence_length - 1]), axis=None) for y in eval_user_sequence_x])


    (day_eval_set_sequence, week_eval_set_sequence, month_eval_set_sequence, season_eval_set_sequence) = eval_periodic_sequence
    dataset = tf.data.Dataset.from_tensor_slices((eval_poi_sequence_x, eval_recent_time_sequence_x,  eval_user_sequence_x, day_eval_set_sequence, week_eval_set_sequence, month_eval_set_sequence, season_eval_set_sequence, eval_poi_seqence_x_out, eval_recent_time_sequence_x_out, eval_user_sequence_x_out, eval_target_dec))


    dataset = dataset.shuffle(buffer_size=len(eval_poi_sequence_x))
    assert batch_size is not None, "eval batchSize must not be None"

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(rearrange)

    dataset = dataset.repeat(1)

    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)


    return iterator.get_next()


def pred_next_string(value,targets):


    index = 0
    for v in value:
        topks = v['topk'] #[[x for x in index] for index in v['topk']]
        print("topk = ", topks)
        print(targets[:,0][index])
        index = index + 1





def mapk(actual, predicted, k):
    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def precisionk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(predicted)


def recallk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(actual)


def ndcgk(actual, predicted):
    idcg = 1.0
    dcg = 1.0 if predicted[0] in actual else 0.0
    for i,p in enumerate(predicted[1:]):
        if p in actual:
            dcg += 1.0 / np.log(i+2)
        idcg += 1.0 / np.log(i+2)
    # print (len(predicted), "dcg = ", dcg, idcg)
    return dcg / idcg