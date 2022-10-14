'''
If you use this code for your research please cite the following paper.
Research Paper: Deep reinforcement learning of dynamic POI generation and optimization for itinerary recommendation
Authors: Sajal Halder, Kwan Hui Lim, Jeffrey Chan and Xiuzhen Zhang,
Implemented By: Sajal Halder
Date: April 2022
Published : Submitted to IEEE Transaction on Knowledge and Discovery (Q1 Rank Journal)
'''

# Import Necessary Libraries
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import numpy as np
import utility as ut
import os
import shutil
# from config import DEFINES
import model2 as mdrl
from sklearn.model_selection import KFold,train_test_split
import timeit

# For warning avoid use following 2 lines
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



class Config(object):
    """ Config."""
    min_poi = 2
    min_user = 2
    batch_size = 32
    train_steps = 1000
    dropout_width = 0.5
    hidden_size = 128
    vocabulary_size = 1000 # POI size default 1000 it may change based on datasets
    learning_rate = 0.001
    shuffle_seek = 100
    max_sequence_length = 25
    embedding_size = 128
    query_dimention  = 128
    key_dimention = 128
    value_dimention = 128
    layers_size = 4
    heads_size = 4
    max_queue_time = 0.0 # default queue time (0)

    check_point_path = "./data_out/check_point"  # We use data_out folder in this directory to save our model

    top_k = 10 # Change this value for top-k results
    number_POIs = 1000 # Default value (1000) it may be changed based on POI number in dataset
    number_User = 1000000 # Default value (1000000) it may be changed based on user number in dataset

    G_P_Q = 1  #  Using GCN or not 1 = ALL DRLIR, 2 = no GCN, 3 = NO periodic pattern, 4 = no Personalization, 5 = no Queueing Time


DEFINES = Config()



def data_Preprocessing(data):
    '''
    :param data: data represents dataset name
    :return: Preprocessed data, avoid unnecessary data and anomaly data using minimum users and minimum POIs in a sequence
    '''

    min_poi, min_user = DEFINES.min_poi, DEFINES.min_user  # User may change this value based on Config implementation

    # Read visits travel sequence
    dfVisits = pd.read_csv("New_Data/userVisits-" + data + "-allPOI.csv", delimiter=",")



    # Remove negative times data as anomaly
    dfVisits = dfVisits[dfVisits.takenUnix > 0]

    # Lebel encoder that converts categorical user id (nsid) to interger userID
    LE = LabelEncoder()
    dfVisits['userID'] = LE.fit_transform(dfVisits['nsid'])
    dfVisits['cateID'] = LE.fit_transform(dfVisits['poiTheme'])

    # Remove un-necessary data. Used only userID, taken time, poiID, poiTheme, cateID, pioFreq and sequence ID
    dfVisits = dfVisits[['userID','takenUnix','poiID','poiTheme','cateID','poiFreq','seqID']]


    # Remove data who have only one sequence and the sequences that has only one POI
    poi_frequence = dfVisits['poiID'].value_counts().reset_index()
    poi_frequence.columns = ['poiID','poiFrequency']
    user_frequency = dfVisits['userID'].value_counts().reset_index()
    user_frequency.columns = ['userID','userFrequency']
    # Update Data sets based on minimum POI and UsersID (default 3)

    merged_Frame = pd.merge(dfVisits, poi_frequence, on='poiID', how='inner')
    dfVisits = merged_Frame[merged_Frame.poiFrequency >= min_poi]
    merged_Frame = pd.merge(dfVisits, user_frequency, on='userID', how='inner')
    dfVisits = merged_Frame[merged_Frame.userFrequency >= min_user]


    sequence_frequency = dfVisits["seqID"].value_counts().reset_index()
    sequence_frequency.columns= ["seqID","seq_frequency"]
    merged_Frame = pd.merge(dfVisits, sequence_frequency, on=["seqID"], how='inner')
    dfVisits = merged_Frame[merged_Frame.userFrequency >= 2]

    dfVisits['poiID'] = dfVisits.poiID.values + 3   # POIID starts from 4 bause first 0-3 ids have been used for special purposes.



    dfVisits = dfVisits[['userID', 'takenUnix', 'poiID', 'poiTheme','cateID', 'poiFreq', 'seqID']]

    # dfVisits = makeSequence_based_6hours(dfVisits, 6*3600)

    dfVisits['seqID'] = LE.fit_transform(dfVisits['seqID'])

    # dfVisits.to_csv("New_Data/userVisits-" + data + "-allPOI.csv")
    return dfVisits

def makeSequence_based_6hours(dfVisits, time_distance):

    users = dfVisits.userID.unique()
    new_data = pd.DataFrame(columns = ['userID', 'takenUnix', 'poiID', 'poiTheme','cateID', 'poiFreq', 'seqID'], dtype = object)
    Sequence_start = 0
    for user in users:
        tempVisits = dfVisits[dfVisits.userID == user]
        tempVisits = tempVisits.sort_values('takenUnix', ascending=True).reset_index()
        new_data.at[new_data.shape[0]] = [tempVisits.iloc[0].userID, tempVisits.iloc[0].takenUnix, tempVisits.iloc[0].poiID, tempVisits.iloc[0].poiTheme,  tempVisits.iloc[0].cateID,  tempVisits.iloc[0].poiFreq,  Sequence_start]
        for i in range(1, len(tempVisits)):
            if tempVisits.iloc[i].takenUnix - tempVisits.iloc[i-1].takenUnix > time_distance:
                Sequence_start = Sequence_start+1

            new_data.at[new_data.shape[0]] = [tempVisits.iloc[i].userID, tempVisits.iloc[i].takenUnix, tempVisits.iloc[i].poiID, tempVisits.iloc[i].poiTheme, tempVisits.iloc[i].cateID, tempVisits.iloc[i].poiFreq, Sequence_start]

        Sequence_start = Sequence_start + 1

    return new_data

def rideDuration(dataset, dfVisits, maxPOINumber):
    '''
    :param dataset: Dataset name
    :return: POIs ride durations
    '''
    # Read Training dataset

    sequences = dfVisits.seqID.unique()
    # maxPOINumber = max(dfVisits.poiID.unique())+1 # Because POI number is started from 1
    # DEFINES.number_POIs = maxPOINumber
    # DEFINES.number_User = max(dfVisits.userID.unique()) + 1
    # Covisitors Matrix and POIs popularity dictionary
    coVisitorsMatrix = np.zeros((maxPOINumber,maxPOINumber))

    POI_popoularity = {}
    POI_rideTime = {}
    POI_frequency = {}
    POI_avgRideTime = {}

    for seq in sequences:
        seqVisits = dfVisits[dfVisits.seqID == seq]    # find sequence
        seqVisits = seqVisits.sort_values('takenUnix', ascending=True).reset_index()
        prePOI = seqVisits.iloc[0].poiID
        preTime = seqVisits.iloc[0].takenUnix
        for i in range(1,len(seqVisits)):
            curPOI = seqVisits.iloc[i].poiID
            if prePOI != curPOI:
                POI_popoularity[prePOI] = 1 if prePOI not in POI_popoularity else POI_popoularity[prePOI] + 1
                # Update covisitor matrix
                coVisitorsMatrix[prePOI][curPOI] = coVisitorsMatrix[prePOI][curPOI] + 1
                coVisitorsMatrix[prePOI][prePOI] = coVisitorsMatrix[prePOI][prePOI] + 1
                # Update travel time
                time_dif = seqVisits.iloc[i-1].takenUnix - preTime
                POI_rideTime[prePOI] = time_dif if prePOI not in POI_rideTime else POI_rideTime[prePOI] + time_dif
                POI_frequency[prePOI] = 1 if prePOI not in POI_frequency else POI_frequency[prePOI] + 1

                preTime = seqVisits.iloc[i].takenUnix

                prePOI = curPOI
    # For last sequence updates
    curPOI = dfVisits.iloc[len(dfVisits) - 1].poiID
    coVisitorsMatrix[prePOI][curPOI] = coVisitorsMatrix[prePOI][curPOI] + 1   # Covisits patters [x][y]
    coVisitorsMatrix[curPOI][curPOI] = coVisitorsMatrix[curPOI][curPOI] + 1     # Popularity values in the matrix [x][x]
    time_dif = dfVisits.iloc[len(dfVisits) - 1].takenUnix - preTime
    POI_rideTime[prePOI] = time_dif if prePOI not in POI_rideTime else POI_rideTime[prePOI] + time_dif
    POI_frequency[prePOI] = 1 if prePOI not in POI_frequency else POI_frequency[prePOI] + 1


    for POI in POI_rideTime.keys():
        POI_avgRideTime[POI] = POI_rideTime[POI]/POI_frequency[POI]

    rideData = pd.DataFrame(POI_avgRideTime.items(), columns=['poiID', 'rideDuration'])

    # Read POIs other information inclusing POI name, category and position
    POI_data = pd.read_csv("Data/POI-"+dataset+".csv",delimiter = ";")
    POI_data = POI_data[['poiID','poiName','lat','long','theme']]
    # Marge POI data with ride calculation data
    margeData = pd.merge(POI_data,rideData,on='poiID', how='inner')
    # Save marge Data

    # Update covisits matrix based on covisits and popularity values
    coVisitorsMatrix = ut.updateMatrix_basedPopularity(coVisitorsMatrix)
    #Construct GCN model from the co-visits patterns
    GCN = ut.gcn_network(coVisitorsMatrix, maxPOINumber, DEFINES.hidden_size, DEFINES.hidden_size)  # We use four special nodes
    #print("GCN Shape= ", GCN.shape)

    margeData.to_excel("Processed_data/POI-"+dataset+".xlsx",index=False)
    # Save  data a plk file
    a_file = open("Processed_data/GCN_"+dataset + ".pkl", "wb")
    pickle.dump(GCN, a_file)
    a_file.close()

    print(dataset +" Dataset Ride Duration Done.")


# Make sequences data
def build_sequencs(dfVisits, travel_sequences):
    '''
    :param dfVisits: all data
    :sequences : seuences (might be training, testing and validation)
    :return: sequences information
    '''

    sequences = []
    time_sequences = []
    user_sequences = []


    for seq in travel_sequences:
        seqVisits = dfVisits[dfVisits.seqID == seq]  # find sequence
        # if sequence contains only one POIs avoid it
        if len(seqVisits.poiID.unique()) < 2:
            continue
        seqVisits = seqVisits.sort_values('takenUnix', ascending=True).reset_index()
        sequence = []
        times = []
        prePOI = seqVisits.iloc[0].poiID
        preTime = seqVisits.iloc[0].takenUnix
        sequence.append(prePOI)
        times.append(preTime)

        user_sequences.append(seqVisits.iloc[0].userID)
        for i in range(1, len(seqVisits)):
            curPOI = seqVisits.iloc[i].poiID

            if prePOI != curPOI:
                curtime = seqVisits.iloc[i].takenUnix
                sequence.append(curPOI)
                times.append(curtime)
                prePOI = curPOI

        sequences.append(sequence)
        time_sequences.append(times)

    return sequences, time_sequences, user_sequences


def build_model_input(POI_sequences,Time_sequences,User_sequences): #dataset,option):
    '''

    :param dataset:
    :param Optoin:
    :return: model sequences for train and level data
    '''
    new_POI_sequences = []
    new_Time_sequences = []
    new_user_sequences = []


    for i in range(len(POI_sequences)):
        sequences = POI_sequences[i]
        time_sequencs = Time_sequences[i]
        # print(" Sequence and time sequence =", len(sequences), len(time_sequencs))
        seq_len = len(sequences)
        for j in range(2,seq_len):
            new_POI_sequences.append(sequences[max(0, j-DEFINES.max_sequence_length):j])
            new_Time_sequences.append(time_sequencs[max(0, j - DEFINES.max_sequence_length):j])
            new_user_sequences.append([User_sequences[i]] * (j- max(0, j - DEFINES.max_sequence_length)))  # Make user sequence based on same user id
    return new_POI_sequences, new_Time_sequences, new_user_sequences

def make_train_level_data(data):
    '''
    :param data: sequence makes train x and lebel y
    :return: x and y which indicate train and lebel data
    '''
    x = [d[0:len(d)-1] for d in data]
    y = [d[len(d) - 1:len(d)] for d in data]
    return x,y

def makeCovisitSequences(GCN, POI_sequences):
    '''
    :param GCN: GCN transition matrix
    :return: covisits sequences based on GCN model
    '''

    GCN_sequences = [[1.0]+ [GCN[int(y[i-1]),int(y[i])] for i in range(1,len(y))] for y in POI_sequences]

    return GCN_sequences

def makePeriodicSequecnes(time_sequences):
    '''
    :param time_sequences: time sequenes
    :return: periodic sequences season, weeks and days
    '''
    season_sequences = []
    month_sequences = []
    weekdays_sequences = []
    day_sequences = []

    for t_seq in time_sequences:
        seasons, month, weekdays, day = ut.Convert_Time_ID(t_seq)
        season_sequences.append(seasons)
        month_sequences.append(month)
        weekdays_sequences.append(weekdays)
        day_sequences.append(day)

    return season_sequences, month_sequences, weekdays_sequences, day_sequences

def train_model(train_input,train_poi_sequence_y,data, GCN):
    # # Design  network
    with tf.Graph().as_default(), tf.compat.v1.Session() as session:

        tf.compat.v1.global_variables_initializer().run()

        if (os.path.exists(DEFINES.check_point_path)):
            shutil.rmtree(DEFINES.check_point_path, ignore_errors=True)

        check_point_path = os.path.join(os.getcwd(), DEFINES.check_point_path)
        os.makedirs(check_point_path, exist_ok=True)

        # Make up an estimator.
        classifier = tf.estimator.Estimator(
            model_fn=mdrl.Model_DRLIR,  # Register the model.
            model_dir=DEFINES.check_point_path,  # Register checkpoint location.
            # model_reward = Rewards,
            params={  # Pass parameters to the model.
                'hidden_size': DEFINES.hidden_size,  # Set the weight size.
                'learning_rate': DEFINES.learning_rate,  # Set learning rate.
                'vocabulary_length': DEFINES.number_POIs,  # Sets the dictionary size.
                'embedding_size': DEFINES.embedding_size,  # Set the embedding size.
                'max_sequence_length': DEFINES.max_sequence_length,
                'user_length': DEFINES.number_User,
                'gcn_values': GCN,
                'batch_size': DEFINES.batch_size,
                'G_P_Q': DEFINES.G_P_Q,
                'heads_size': DEFINES.heads_size,
                'layers_size': DEFINES.layers_size,
                'top_k': DEFINES.top_k
            })

        classifier.train(input_fn=lambda: mdrl.train_input_fn(train_input, train_poi_sequence_y.astype('int64'), DEFINES.batch_size), steps=DEFINES.train_steps)

        # Save the model after training
        # Save  time sequence data a plk file
        a_file = open("Model/Model_" + data + ".pkl", "wb")
        pickle.dump(classifier, a_file)
        a_file.close()

def evaluate_model(validation_input,validation_poi_sequence_y,data):
    # # validation results
    with open("Model/Model_" + data + ".pkl", "rb") as f:
        classifier = pickle.load(f)

    eval_result = classifier.evaluate(input_fn=lambda: mdrl.eval_input_fn(validation_input, validation_poi_sequence_y.astype('int64'), DEFINES.batch_size), steps=1)

    c_pre_5, c_recall_5, c_f1_5, c_pre_10, c_recall_10, c_f1_10, c_ndcg_5, c_ndcg_10 = '{precision_5:0.3f},{recall_5:0.3f}, {f1_5:0.3f}, {precision_10:0.3f},{recall_10:0.3f},{f1_10:0.3f}, {ndcg_5:0.3f},{ndcg_10:0.3f}'.format(
        **eval_result).split(",")

    print(data + " validation results = ", c_pre_5, c_recall_5, c_f1_5, c_ndcg_5, c_pre_10, c_recall_10, c_f1_10,
          c_ndcg_10)

def test_model(test_input, test_poi_sequence_y, data):
    # # validation results
    with open("Model/Model_" + data + ".pkl", "rb") as f:
        classifier = pickle.load(f)
    eval_result = classifier.evaluate(input_fn=lambda: mdrl.eval_input_fn(test_input, test_poi_sequence_y.astype('int64'), DEFINES.batch_size), steps=1)

    c_pre_5, c_recall_5, c_f1_5, c_ndcg_5, c_pre_10, c_recall_10, c_f1_10,  c_ndcg_10 = '{precision_5:0.3f},{recall_5:0.3f}, {f1_5:0.3f},  {ndcg_5:0.3f}, {precision_10:0.3f},{recall_10:0.3f},{f1_10:0.3f},{ndcg_10:0.3f}'.format( **eval_result).split(",")


    return c_pre_5, c_recall_5, c_f1_5, c_ndcg_5, c_pre_10, c_recall_10, c_f1_10, c_ndcg_10

def build_queue_times(data):
    # Read Queuing time files

    queueInfo = pd.read_csv("Data/queueTimes-" + data + ".csv",delimiter=';')
    # Normalize the queuing time
    maxQueue = max(queueInfo['avgQueueTime'].values)

    queueInfo['poiID'] = queueInfo.poiID.values + 3  # Make staring point 4

    return queueInfo,maxQueue

def make_itinerary_priodic_sequence(temp_time_seq):
    temp_season_sequence, temp_month_sequence, temp_weekday_sequence, temp_day_sequence = makePeriodicSequecnes(temp_time_seq)

    temp_season_sequence = ut.padding(temp_season_sequence, DEFINES.max_sequence_length)
    temp_month_sequence = ut.padding(temp_month_sequence, DEFINES.max_sequence_length)
    temp_weekday_sequence = ut.padding(temp_weekday_sequence, DEFINES.max_sequence_length)
    temp_day_sequence = ut.padding(temp_day_sequence, DEFINES.max_sequence_length)

    temp_periodic_sequence = [temp_day_sequence, temp_weekday_sequence, temp_month_sequence, temp_season_sequence]

    return temp_periodic_sequence


def next_move(data, budget, temp_POI_seq, reco_seq_id, temp_time_seq, temp_user_seq,temp_poi_sequence_y,costInfo, QueueTimes, maxQueue, startHour,pre_item,endNode):

    recent_time_sequences = ut.padding(ut.makeRecentSequenceOne(temp_time_seq), DEFINES.max_sequence_length)
    temp_POI_seq = ut.padding(temp_POI_seq, DEFINES.max_sequence_length)
    temp_time_seq = ut.padding(temp_time_seq, DEFINES.max_sequence_length)
    temp_user_seq = ut.padding(temp_user_seq, DEFINES.max_sequence_length)

    temp_periodic_sequence = make_itinerary_priodic_sequence(temp_time_seq)



    temp_input = [temp_POI_seq, recent_time_sequences, temp_user_seq, temp_periodic_sequence]

    # # Import classifier Model model
    with open("Model/Model_" + data + ".pkl", "rb") as f:
        classifier = pickle.load(f)


    predict_result = classifier.predict(input_fn=lambda: mdrl.eval_input_fn(temp_input, temp_poi_sequence_y.astype('int64'),DEFINES.batch_size))  # DEFINES.batch_size))  # Here batch size = 1

    update_topks = []
    update_values = []
    topk = DEFINES.top_k  # Candidate set number 5 or 10 based on config values
    index = 0
    for v in predict_result:
        top_k = v['topk']#[0]
        top_k_values = v['reward']#[0]
        existing_POIs = [int(item) for item in temp_POI_seq[index] if item > 0]

        top_k_items = [i for i in top_k if i > 3 and i not in existing_POIs]
        top_k_items_values = [top_k_values[i] for i in top_k if i > 3 and i not in existing_POIs]
        top_k_items = top_k_items[0:topk]


        top_k_items_values = top_k_items_values[0:topk]
        index = index + 1
        update_topks.append(top_k_items)
        update_values.append(top_k_items_values)

        if index == len(temp_POI_seq):
            break

    if len(update_topks[0]) < 1:
        return [], [], [], []

    rideTime, queueTime, travelTime, = calculate_R_Q_T_TIME( costInfo, QueueTimes, update_topks, startHour, pre_item, len(reco_seq_id))

    update_poi_list = []
    for j in range(len(reco_seq_id)):
        current_pois = update_topks[j]
        current_values = update_values[j]
        current_queuetime = queueTime[j]
        # If G_P_Q == 5 that means no queuing impact for next POI selection
        if DEFINES.G_P_Q != 5:
            update_values_with_queue = [current_values[i]/(current_queuetime[i]/maxQueue) for i in range(len(current_values))]
            sort_index = (-np.asarray(update_values_with_queue).argsort())
            new_list = [current_pois[i] for i in sort_index]
        else:
            new_list = current_pois

        update_poi_list.append([new_list[0:3]])

    rideTime_one, queueTime_one, travelTime_one, next_POIs = calculate_R_Q_T_TIME_top3(budget, costInfo, QueueTimes, update_poi_list, startHour, pre_item, len(reco_seq_id),endNode)
    # rideTime_one, queueTime_one, travelTime_one = calculate_R_Q_T_TIME( costInfo, QueueTimes,update_poi_list, startHour,pre_item, len(reco_seq_id))
    # print("Next POIs = ", next_POIs)
    # print(" update poi list = ", update_poi_list)
    totalTime = [rideTime_one[i] + travelTime_one[i] + queueTime_one[i] for i in range(len(reco_seq_id))]
    temp_budget = np.asarray([budget[i] - totalTime[i] for i in range(len(reco_seq_id))])

    b_index = [True if temp_budget[i] > 0 else False for i in range(len(reco_seq_id))] # Maximum budget 24 hours  ( 24*36000 = 85400)
    # print(" len reco seq id =", len(reco_seq_id), reco_seq_id)
    update_reco_seq_id = np.asarray(reco_seq_id)[b_index]
    # print(" len update_reco_seq_id =", len(update_reco_seq_id), update_reco_seq_id)
    next_POIs = np.asarray(next_POIs)[b_index]


    return update_reco_seq_id,next_POIs, totalTime #, temp_budget


def test_Model_Itinerary(data, dfVisits,test_seq ):


    POI_sequences, Time_sequences, User_sequences = build_sequencs(dfVisits, test_seq)

    QueueTimes, maxQueue = build_queue_times(data)


    costInfo = pd.read_csv("Data/costProfCat-"+data+"-all.csv", delimiter=";")
    costInfo['from'] = costInfo['from'].values + 3  # Ensure it start form 4
    costInfo['to'] = costInfo['to'].values + 3 # Ensuere it start from 4


    Original_Seqeunces =  range(len(POI_sequences))
    Recommended_List = {}


    for i in Original_Seqeunces:
        Recommended_List[i] = [POI_sequences[i][0]]

    reco_seq_id = range(len(Original_Seqeunces))


    reco_seq_id_padding = range((len(reco_seq_id) // DEFINES.batch_size + 1) * DEFINES.batch_size)

    temp_POI_seq = [[POI_sequences[x][0]]  for x in reco_seq_id]
    temp_time_seq = [[Time_sequences[x][0]] for x in reco_seq_id]
    temp_user_seq = [[User_sequences[x]] for x in reco_seq_id]

    end_POIs = [[POI_sequences[x][-1]]  for x in reco_seq_id]
    budget = [Time_sequences[x][-1] - Time_sequences[x][0] for x in reco_seq_id]

    Total_budgets = budget


    startTime = [pd.to_datetime(Time_sequences[x][0], unit='s') for x in reco_seq_id]
    startHour = [startTime[x].hour for x in range(len(startTime))]
    endNodes = end_POIs
    print(" End nodes = ", endNodes)
    # print("Max Budget = ", max(budget))
    # print(len(budget), " budget time = ", budget)

    while(len(Original_Seqeunces)>0):

        temp_POI_seq = [temp_POI_seq[x] if x in reco_seq_id else [0] for x in reco_seq_id_padding]
        pre_item = [x[-1] for x in temp_POI_seq]
        temp_time_seq = [temp_time_seq[x] if x in reco_seq_id else [0] for x in reco_seq_id_padding]
        temp_user_seq = [[temp_user_seq[x]] if x in reco_seq_id else [0] for x in reco_seq_id_padding]

        budget = [budget[x] if x in reco_seq_id else 0 for x in reco_seq_id_padding]
        startHour = [startHour[x] if x in reco_seq_id else 0 for x in reco_seq_id_padding]

        temp_poi_sequence_y = ut.padding([[POI_sequences[Original_Seqeunces[x]][-1]] if x in reco_seq_id else [0] for x in reco_seq_id_padding],DEFINES.max_sequence_length)

        update_reco_seq_id, update_poi_list, totalTime = next_move( data, budget, temp_POI_seq, reco_seq_id, temp_time_seq,temp_user_seq, temp_poi_sequence_y, costInfo, QueueTimes, maxQueue, startHour, pre_item,endNodes)
        # print("update list = ", update_poi_list)
        if len(update_reco_seq_id) < 1:
            break

        sequence_ends = []

        for i in range(len(update_reco_seq_id)):
            if update_poi_list[i] > 0 :
                Recommended_List[update_reco_seq_id[i]] = [item for item in Recommended_List[update_reco_seq_id[i]]] + [update_poi_list[i]]
            if update_poi_list[i] == end_POIs[i] and update_poi_list[i] == -1:
                sequence_ends.append(i)


        Original_Seqeunces = [Original_Seqeunces[item] for item in update_reco_seq_id if Original_Seqeunces[item] not in sequence_ends]

        reco_seq_id = range(len(Original_Seqeunces))
        # Updated items for next itireations
        reco_seq_id_padding = range((len(reco_seq_id) // DEFINES.batch_size + 1) * DEFINES.batch_size)

        temp_POI_seq = [[item for item in temp_POI_seq[reco_seq_id[x]]] + [update_poi_list[x]] for x in range(len(reco_seq_id))]
        # emp_POI_seq = [Recommended_List[x] for x in reco_seq_id]
        # print("temp_POI seq = ", temp_POI_seq)
        temp_time_seq = [[item for item in temp_time_seq[x] ]+ [int(temp_time_seq[x][-1]+totalTime[x])] for x in reco_seq_id]
        temp_user_seq = [temp_user_seq[x][0] for x in reco_seq_id] # np.asarray([[temp_user_seq[x][0][0] for i in range(seq_len)] for x in reco_seq_id])

        budget = [budget[x] - totalTime[x] for x in reco_seq_id]
        startTime = [pd.to_datetime(temp_time_seq[x][-1], unit='s') for x in reco_seq_id]
        startHour = [startTime[x].hour for x in reco_seq_id]
        endNodes = [endNodes[x] for x in reco_seq_id]
        #budget = [temp_budget[x] for x in reco_seq_id]

        # print(len(budget), " budget time = ", budget)




    results_itinerary = pd.DataFrame(columns=['sequence','budget', 'precision','recall','f1score','pairf1','ndcg'], dtype = object)
    Recommended_POIS_List = [Recommended_List[x] for x in Recommended_List.keys()]
    sequences = POI_sequences

    for seq in range(len(sequences)):

        #print("Budget = ", Total_budgets[seq])
        original = list(dict.fromkeys(sequences[seq]))
        #print("Original list = ", original)
        #print("Recommended list = ", Recommended_POIS_List[seq])


        precision = precisionk(original, Recommended_POIS_List[seq])  # We avoid first item because it always starts
        recall = recallk(original,Recommended_POIS_List[seq])
        f1_score = 2*precision*recall / (precision + recall + 1e-8)
        pair_fscore = Pairs_F1(original, Recommended_POIS_List[seq])
        ndcg = ndcgk(sequences[seq],Recommended_POIS_List[seq])
        # print(precision, recall, f1_score, pair_fscore, ndcg)

        results_itinerary.at[results_itinerary.shape[0]] = [seq,Total_budgets[seq],precision,recall,f1_score, pair_fscore, ndcg]

    P, R, F1, P_F1, N = np.mean(results_itinerary.precision),np.mean(results_itinerary.recall),np.mean(results_itinerary.f1score), np.mean(results_itinerary.pairf1),np.mean(results_itinerary.ndcg)
    print("Aaverage Resutls = ", P, R, F1, P_F1, N )
    return P, R, F1, P_F1, N


def calculate_R_Q_T_TIME_top3(budetTime, costInfo, QueueTimes, update_topks, startHour,pre_item, sequence_len,endNode):
    rideTime = []
    queueTime = []
    travelTime = []
    next_POIs = []
    endNode = [item [0] for item in endNode]
    for i in range(sequence_len):
        #print("items = ", update_topks[i][0])
        # print("end node = ", endNode[i])

        r =  np.asarray([costInfo[costInfo['to'] == item].rideDuration.values[0] for item in update_topks[i][0]])
        t = np.asarray([costInfo[(costInfo['from'] == pre_item[i]) & (costInfo['to'] == item)].walkTime.values[0] if pre_item[i] != item else 1.0 for item in update_topks[i][0]] ) # We want to avoid same node selectin thus it makes long distance 1000000
        # print(" r = ", r)
        # print("t = ", t)
        hours = np.asarray([(startHour[i] + (r[j] + t[j])// 3600) % 24 for j in range(len(update_topks[i][0]))])
        # print(" hour = ", hours)
        # hour = [(startHour[i] + (r[j]+t[j])//3600)%24 for j in range(len(update_topks[i][0]))]
        q = [QueueTimes[(QueueTimes.poiID == update_topks[i][0][j]) & (QueueTimes.hour == hours[j])].avgQueueTime.values[0] for j in range (len(update_topks[i][0]))]


        # check travel time, visiting time and queuing time from selected node to destination node

        r_to_end = np.asarray([costInfo[costInfo['to'] == endNode[i]].rideDuration.values[0] for item in update_topks[i][0]])
        #print("r_to_end = ", r_to_end)
        t_to_end = np.asarray([costInfo[(costInfo['from'] == item) & (costInfo['to'] == endNode[i])].walkTime.values[0] if item != endNode[i] else 1.0 for item in update_topks[i][0]])  # We want to avoid same node selectin thus it makes long distance 1000000
        #print(" t_to_end = ",t_to_end)
        q_to_end = [QueueTimes[(QueueTimes.poiID == endNode[i]) & (QueueTimes.hour == hours[j])].avgQueueTime.values[
                0] for j in range(len(update_topks[i][0]))]

        #print("q_to_end = ", q_to_end)
        if (r[0]+q[0]+t[0] + r_to_end[0] +  t_to_end [0] + q_to_end[0] <= budetTime[i]):
            rideTime.append(r[0])
            queueTime.append(q[0])
            travelTime.append(t[0])
            next_POIs.append(update_topks[i][0][0])
                # break
        elif (r[1]+q[1]+t[1] + r_to_end[1] +  t_to_end [1] + q_to_end [1] <= budetTime[i]):
            rideTime.append(r[1])
            queueTime.append(q[1])
            travelTime.append(t[1])
            next_POIs.append(update_topks[i][0][1])
        elif (r[2]+q[2]+t[2]  + r_to_end[2] +  t_to_end [2] + q_to_end [2] <= budetTime[i]):
            rideTime.append(r[2])
            queueTime.append(q[2])
            travelTime.append(t[2])
            next_POIs.append(update_topks[i][0][2])
        else:
            rideTime.append(100000)
            queueTime.append(100000)
            travelTime.append(100000)
            next_POIs.append(endNode[i])



    return rideTime, queueTime, travelTime, next_POIs


def calculate_R_Q_T_TIME(costInfo, QueueTimes, update_topks, startHour, pre_item, sequence_len):


    rideTime = np.asarray([[costInfo[costInfo['to'] == item].rideDuration.values[0] for item in update_topks[i]] for i in range(sequence_len)])
    travelTime = np.asarray([[costInfo[(costInfo['from'] == pre_item[i]) & (costInfo['to'] == item)].walkTime.values[0] if pre_item[i] != item else 1000000 for item in update_topks[i]] for i in range(sequence_len)])  # We want to avoid same node selectin thus it makes long distance 1000000
    hours = np.asarray([[(startHour[i] + (rideTime[i][j] + travelTime[i][j])//3600)%24 for j in range(len(update_topks[i]))]for i in range(sequence_len)])
    # print("start Hour = ", startHour)
    # print("rideTime = ", rideTime.shape, rideTime)
    # print(" travel time = ", travelTime.shape, travelTime)
    # print("Hours = ", hours.shape, hours)
    queueTime = np.asarray([[QueueTimes[(QueueTimes.poiID == update_topks[i][j]) & (QueueTimes.hour == hours[i][j])].avgQueueTime.values[0] for j in
         range(len(update_topks[i]))] for i in range(sequence_len)])
    # print("queueTime = ", queueTime)
    return rideTime, queueTime, travelTime

def Pairs_F1(original, predict):

    # Make unique ids to execute pairs value
    mainPOIs = set(original).union(set(predict))


    newPOIsNumber = {}
    index = 0
    for i in mainPOIs:
        if i not in newPOIsNumber:
            newPOIsNumber[i] = index
            index = index + 1

    original = [newPOIsNumber[i] for i in original]
    predict = [newPOIsNumber[i] for i in predict]


    max_POI = len(newPOIsNumber)

    matrix = np.zeros((max_POI, max_POI))
    matrix2 = np.zeros((max_POI, max_POI))

    for i in range(len(original) - 1):
        for j in range(i + 1, len(original)):
            matrix[original[i], original[j]] = 1

    for i in range(len(predict) - 1):
        for j in range(i + 1, len(predict)):
            matrix2[predict[i], predict[j]] = 1

    counts = 0
    count_rec = 0
    count_ori = 0

    for i in range(max_POI):
        for j in range(max_POI):
            if matrix[i, j] == 1 and (matrix[i, j] == matrix2[i, j]):
                counts = counts + 1
            if matrix[i, j] == 1:
                count_ori = count_ori + 1
            if matrix2[i, j] == 1:
                count_rec = count_rec + 1

    precision = counts / (count_rec + 1e-8)
    recall = counts / (count_ori + 1e-8)
    pair_f1_score = 2 * precision * recall / (precision + recall + 1e-8)

    return pair_f1_score

def mapk(actual, predicted, k):
    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual)+1e-8, k)

def precisionk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / (len(predicted)+1e-8)


def recallk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / (len(actual)+1e-8)


def ndcgk(actual, predicted):
    idcg = 1.0
    dcg = 1.0 if predicted[0] in actual else 0.0
    for i,p in enumerate(predicted[1:]):
        if p in actual:
            dcg += 1.0 / np.log(i+2)
        idcg += 1.0 / np.log(i+2)
    return dcg / idcg

def makeInputs(dataset, dfVisits, train_seq, valid_seq, test_seq, maxPOINumber):

    rideDuration(dataset, dfVisits,maxPOINumber)

    train_poi_sequences, train_time_sequences, train_user_sequence = build_sequencs(dfVisits, train_seq)
    validation_poi_sequences, validation_time_sequences, validation_user_sequence = build_sequencs(dfVisits,valid_seq)
    test_poi_sequences, test_time_sequences, test_user_sequence = build_sequencs(dfVisits, test_seq)

    train_poi_sequences, train_time_sequences, train_user_sequence = build_model_input(train_poi_sequences, train_time_sequences, train_user_sequence )
    validation_poi_sequences, validation_time_sequences, validation_user_sequence = build_model_input(validation_poi_sequences, validation_time_sequences, validation_user_sequence)
    test_poi_sequences, test_time_sequences, test_user_sequence = build_model_input(test_poi_sequences, test_time_sequences, test_user_sequence)

    train_poi_sequence_x, train_poi_sequence_y = make_train_level_data(train_poi_sequences)
    validation_poi_sequence_x, validation_poi_sequence_y = make_train_level_data(validation_poi_sequences)
    test_poi_sequence_x, test_poi_sequence_y = make_train_level_data(test_poi_sequences)

    train_time_sequence_x, train_time_sequence_y = make_train_level_data(train_time_sequences)
    validation_time_sequence_x, validation_time_sequence_y = make_train_level_data(validation_time_sequences)
    test_time_sequence_x, test_time_sequence_y = make_train_level_data(test_time_sequences)

    train_user_sequence_x, train_user_sequence_y = make_train_level_data(train_user_sequence)
    validation_user_sequence_x, validation_user_sequence_y = make_train_level_data(validation_user_sequence)
    test_user_sequence_x, test_user_sequence_y = make_train_level_data(test_user_sequence)

    train_recent_time_sequence_x, validation_recent_time_sequence_x, test_recent_time_sequence_x = ut.makeRecentSequence(train_time_sequence_x, validation_time_sequence_x, test_time_sequence_x)


    train_season_sequence, train_month_sequence, train_weekday_sequence, train_day_sequence = makePeriodicSequecnes(train_time_sequence_x)
    validation_season_sequence, validation_month_sequence, validation_weekday_sequence, validation_day_sequence = makePeriodicSequecnes(validation_time_sequence_x)
    test_season_sequence, test_month_sequence, test_weekday_sequence, test_day_sequence = makePeriodicSequecnes(test_time_sequence_x)

    # make same length based sequences
    train_poi_sequence_x = ut.padding(train_poi_sequence_x, DEFINES.max_sequence_length)
    validation_poi_sequence_x = ut.padding(validation_poi_sequence_x, DEFINES.max_sequence_length)
    test_poi_sequence_x = ut.padding(test_poi_sequence_x, DEFINES.max_sequence_length)

    train_recent_time_sequence_x = ut.padding(train_recent_time_sequence_x, DEFINES.max_sequence_length)
    validation_recent_time_sequence_x = ut.padding(validation_recent_time_sequence_x, DEFINES.max_sequence_length)
    test_recent_time_sequence_x = ut.padding(test_recent_time_sequence_x, DEFINES.max_sequence_length)

    train_user_sequence_x = ut.padding(train_user_sequence_x, DEFINES.max_sequence_length)
    validation_user_sequence_x = ut.padding(validation_user_sequence_x, DEFINES.max_sequence_length)
    test_user_sequence_x = ut.padding(test_user_sequence_x, DEFINES.max_sequence_length)

    train_season_sequence = ut.padding(train_season_sequence, DEFINES.max_sequence_length)
    validation_season_sequence = ut.padding(validation_season_sequence, DEFINES.max_sequence_length)
    test_season_sequence = ut.padding(test_season_sequence, DEFINES.max_sequence_length)

    train_month_sequence = ut.padding(train_month_sequence, DEFINES.max_sequence_length)
    validation_month_sequence = ut.padding(validation_month_sequence, DEFINES.max_sequence_length)
    test_month_sequence = ut.padding(test_month_sequence, DEFINES.max_sequence_length)

    train_weekday_sequence = ut.padding(train_weekday_sequence, DEFINES.max_sequence_length)
    validation_weekday_sequence = ut.padding(validation_weekday_sequence, DEFINES.max_sequence_length)
    test_weekday_sequence = ut.padding(test_weekday_sequence, DEFINES.max_sequence_length)

    train_day_sequence = ut.padding(train_day_sequence, DEFINES.max_sequence_length)
    validation_day_sequence = ut.padding(validation_day_sequence, DEFINES.max_sequence_length)
    test_day_sequence = ut.padding(test_day_sequence, DEFINES.max_sequence_length)

    train_periodic_sequence = [train_day_sequence, train_weekday_sequence, train_month_sequence, train_season_sequence]
    validation_periodic_sequence = [validation_day_sequence, validation_weekday_sequence, validation_month_sequence, validation_season_sequence]
    test_periodic_sequence = [test_day_sequence, test_weekday_sequence, test_month_sequence, test_season_sequence]

    train_input = [train_poi_sequence_x, train_recent_time_sequence_x, train_user_sequence_x, train_periodic_sequence]
    validation_input = [validation_poi_sequence_x, validation_recent_time_sequence_x,  validation_user_sequence_x, validation_periodic_sequence]
    test_input = [test_poi_sequence_x, test_recent_time_sequence_x, test_user_sequence_x, test_periodic_sequence]

    train_poi_sequence_y = ut.padding(train_poi_sequence_y, DEFINES.max_sequence_length)
    validation_poi_sequence_y = ut.padding(validation_poi_sequence_y, DEFINES.max_sequence_length)
    test_poi_sequence_y = ut.padding(test_poi_sequence_y, DEFINES.max_sequence_length)

    return train_input, validation_input,test_input, train_poi_sequence_y, validation_poi_sequence_y,test_poi_sequence_y


def main(data):

    Fold_Number = 5
    kf = KFold(n_splits=Fold_Number)   # Five fold cross validation

    train_exe_times = []
    test_exe_times = []
    valid_exe_times = []
    itinerary_exe_times = []
    current_time1 = timeit.default_timer()



    Itinerary_results = pd.DataFrame(columns=['precision','recall','f1_score','pari_f1score','ndcg'], dtype=object)
    results = pd.DataFrame(columns=['pre_5', 'recall_5', 'f1_5', 'ndcg_5', 'pre_10', 'recall_10', 'f1_10', 'ndcg_10'], dtype=object)

    dfVisits = data_Preprocessing(data)



    maxPOINumber = min(max(dfVisits.poiID.unique()) + 1, DEFINES.number_POIs)
    maxUserNumber = min(max(dfVisits.userID.unique()) + 1, DEFINES.number_User)

    DEFINES.number_POIs = maxPOINumber
    DEFINES.number_User = maxUserNumber

    dfVisits = dfVisits[dfVisits.poiID < DEFINES.number_POIs]
    dfVisits = dfVisits[dfVisits.userID < DEFINES.number_User]




    sequences  = dfVisits.seqID.unique()
    #print("Sequence Len = ", len(sequences))

    for train_validation, test in kf.split(range(len(sequences))):
        t_v_data_range = range(len(train_validation))
        t, v = train_test_split(t_v_data_range, test_size= 0.12, random_state=42)
        train = [train_validation[i] for i in t]
        valid = [train_validation[i] for i in v]

        train_seq = [sequences[i] for i in train]
        test_seq = [sequences[i] for i in test]
        valid_seq = [sequences[i] for i in valid]

        #print("train seq len = ", len(train_seq), " valid seq = ", len(valid_seq) , " test seq = ", len(test_seq))


        train_input, validation_input, test_input, train_out_label, validation_out_label, test_out_label = makeInputs(data, dfVisits, train_seq, valid_seq, test_seq, DEFINES.number_POIs)


        with open("Processed_data/GCN_" + data + ".pkl", "rb") as f:
            GCN = pickle.load(f)


        # try:
            #
        train_model(train_input,train_out_label,data,GCN)

        current_time2 = timeit.default_timer()
        train_exe_times.append(current_time2-current_time1)

        # # Validate Model
        evaluate_model(validation_input,validation_out_label,data)
        #
        current_time3 = timeit.default_timer()
        valid_exe_times.append(current_time3 - current_time2)



        # # test Model for POI Recommendation
        c_pre_5, c_recall_5, c_f1_5, c_ndcg_5, c_pre_10, c_recall_10, c_f1_10, c_ndcg_10 = test_model(test_input, test_out_label, data)
        print(data + " test results = ", c_pre_5, c_recall_5, c_f1_5, c_ndcg_5, c_pre_10, c_recall_10, c_f1_10,c_ndcg_10)
        results.at[results.shape[0]] = [c_pre_5, c_recall_5, c_f1_5, c_ndcg_5, c_pre_10, c_recall_10, c_f1_10, c_ndcg_10]
        current_time4= timeit.default_timer()
        test_exe_times.append(current_time4 - current_time3)

        # Test Model for Itinerary Recommendation


        precision, recall, f1score, pair_f1score, ndcg = test_Model_Itinerary(data,dfVisits, test_seq)
        print("Itinerary scores = ",  precision, recall,f1score, pair_f1score, ndcg)
        Itinerary_results.at[Itinerary_results.shape[0]] = [precision, recall,f1score, pair_f1score, ndcg]

        current_time5 = timeit.default_timer()
        itinerary_exe_times.append(current_time5 - current_time4)

        current_time1 = timeit.default_timer()
        # except:
        #     pass


    results.to_excel("Results_DLIR/DRLIR_"+data+ "_"+ str(DEFINES.G_P_Q) + ".xlsx")
    Itinerary_results.to_excel("Results_DLIR/DLIR_Itinerary_" + data + ".xlsx")
    print(data , " Final Results = ", np.mean(Itinerary_results.precision),np.mean(Itinerary_results.recall),np.mean(Itinerary_results.f1_score),np.mean(Itinerary_results.pari_f1score), np.mean(Itinerary_results.ndcg))

    train_exe_times = np.asarray(train_exe_times)
    valid_exe_times = np.asarray(valid_exe_times)
    test_exe_times = np.asarray(test_exe_times)
    itinerary_exe_times = np.asarray(itinerary_exe_times)

    print(" Results = ", np.sum(train_exe_times) / Fold_Number, np.sum(valid_exe_times) / Fold_Number, np.sum(test_exe_times) / Fold_Number, np.sum(itinerary_exe_times) / Fold_Number)

    return np.sum(train_exe_times) / Fold_Number, np.sum(valid_exe_times) / Fold_Number, np.sum(test_exe_times) / Fold_Number, np.sum(itinerary_exe_times) / Fold_Number


if __name__ == '__main__':

    Execution_time = pd.DataFrame(columns=['Dataset', 'C_P_G','train_time','valid_time','test_time', 'itinerary_time','total_time'],dtype=object)
    datasets = ['disHolly','epcot','caliAdv','MagicK','Buda','Edin','Toro','Melbourne']


    for data in datasets:
        for CPG in range(1, 2):
            DEFINES.G_P_Q = CPG
            print(data)
            start_time = timeit.default_timer()
            train_time, valid_time, test_time, itinerary_time = main(data)
            print("Times =", train_time, valid_time, test_time, itinerary_time)
            end_time = timeit.default_timer()

            print("Total itinerary Time = ", end_time-start_time)
            Execution_time.at[Execution_time.shape[0]] = [data, DEFINES.G_P_Q, train_time, valid_time, test_time, itinerary_time,end_time-start_time]


    Execution_time.to_excel('Results_DLIR/Execution_Time_DRLIR_results.xlsx')