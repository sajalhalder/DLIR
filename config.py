# -*- coding: utf-8 -*-
import tensorflow as tf

tf.compat.v1.flags.DEFINE_integer('batch_size', 32, 'batch size')  # batch size
tf.compat.v1.flags.DEFINE_integer('train_steps', 200, 'train steps')  # learning epoch
tf.compat.v1.flags.DEFINE_float('dropout_width', 0.5, 'dropout width')  # Dropout size
tf.compat.v1.flags.DEFINE_integer('topk', 5, 'topk')  # Dropout size
tf.compat.v1.flags.DEFINE_float('alpha', 1.0, 'alpha')  # Dropout size

tf.compat.v1.flags.DEFINE_integer('hidden_size', 128, 'weights size')  # weight size # use paper 512
tf.compat.v1.flags.DEFINE_integer('vocabulary_size', 0, 'vocab size')  # weight size # use paper 512
tf.compat.v1.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')  # Learning rate
tf.compat.v1.flags.DEFINE_integer('shuffle_seek', 100, 'shuffle random seek')  # shuffle seed value
tf.compat.v1.flags.DEFINE_integer('max_sequence_length', 25, 'max sequence length')  # Sequence length
tf.compat.v1.flags.DEFINE_integer('embedding_size', 128,'embedding size')  # embedding size # paper 512 using learning speed and performance tuning
tf.compat.v1.flags.DEFINE_integer('query_dimention', 128, 'q # uery dimention')  # Paper 512 Use Learning Speed ​​and Performance Tuning
tf.compat.v1.flags.DEFINE_integer('key_dimention', 128, 'key dimention')  # Use paper 512 Learning speed and performance tuning
tf.compat.v1.flags.DEFINE_integer('value_dimention', 128,'value dimention')  # Paper 512 Use Learning Speed ​​and Performance Tuning
tf.compat.v1.flags.DEFINE_integer('layers_size', 2, 'layers size')  # Use 6 layers or 2 papers to tune learning speed and performance
tf.compat.v1.flags.DEFINE_integer('heads_size', 4,'heads size')  # Papers use 8 headers or 4 learning speed and performance tuning
tf.compat.v1.flags.DEFINE_string('data_path', './data_in/input.csv', 'data path')  # Data location
tf.compat.v1.flags.DEFINE_string ('DATA_OUT_PATH', './data_out/', 'data output path') # Data location
tf.compat.v1.flags.DEFINE_string('vocabulary_path', './data_out/vocabularyData.voc', 'vocabulary path')  # Dictionary location
tf.compat.v1.flags.DEFINE_string('check_point_path', './data_out/check_point', 'check point path')  # Checkpoint location
tf.compat.v1.flags.DEFINE_boolean('tokenize_as_morph', False, 'set morph tokenize')  # Whether or not tokenizing is used depending on the morpheme
tf.compat.v1.flags.DEFINE_boolean('conv_1d_layer', True, 'set conv 1d layer')  # Whether to use the second conv1d among the two methods of the paper
tf.compat.v1.flags.DEFINE_boolean('xavier_embedding', True,  'set init xavier embedding')  # Enable or disable embedding using Xavier initialization
tf.compat.v1.flags.DEFINE_boolean('mask_loss', False, 'set masking loss')  # Using masking for loss (PAD, END)
tf.compat.v1.flags.DEFINE_integer('max_queue_time', 0, 'max_queue_time')  # Using masking for loss (PAD, END)

tf.compat.v1.flags.DEFINE_float('test_percentage', 0.3, 'test percentage')  # test part
tf.compat.v1.flags.DEFINE_float('level_percentage', 0.3, 'level part') # level part
tf.compat.v1.flags.DEFINE_integer('top_k',10, 'top_k') # Use 5 or 10 as top-k values
tf.compat.v1.flags.DEFINE_string('steps','multi_steps', 'multi_steps or single_step') # multi steps or single step
tf.compat.v1.flags.DEFINE_integer('GCN_use', 1, 'GCN_uses')  # Using GCN or not 1 = ALL DRLIR, 2 = no GCN, 3 = NO periodic pattern, 4 = no Queue, 5 = no budget
tf.compat.v1.flags.DEFINE_integer('number_POIs',1000, 'number of POIs')  # Number of POIs
tf.compat.v1.flags.DEFINE_integer('number_User',0, 'number of User')  # Number of POIs


tf.compat.v1.flags.DEFINE_string('dataset','MagicK', 'dataset')  # Datasets: disHolly, epcot, MagicK, caliAdv, disland, Buda, Edin, Toro, Melbourne, Foursquare, Gowalla

# Define FLAGS
DEFINES = tf.compat.v1.flags.FLAGS
