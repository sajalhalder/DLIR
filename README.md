# DLIR : Deep Learning of Dynamic POI Generation and Optimisation for Itinerary Recommendation

In this paper, we propose a deep learning model that addresses itinerary recommendations in a holistic approach that captures user dynamic interest and non-linear spatial dependencies. The proposed model performs in two steps, where the candidate selection policy generates a set of personalised candidate POIs and the itinerary construction step maximises user interest within budget time. To recommend an appropriate candidate set, we propose a multi-head, attention-based transformer to leverage periodic, trends and recent activities to capture user dynamic preferences. We also introduce new co-visiting patterns-based graph convolutional network (GCN) model to capture user non-linear spatial dependencies. To construct the full itinerary from the dynamic candidate sets, we apply greedy policy that incrementally constructs itineraries within the budget time, which aims to maximise user interest and minimise queuing time. Experimental results show that the proposed deep learning model outperforms state-of-the-art baselines in itinerary recommendation and next POI recommendation on eight real datasets.

To use this code in your research work please cite the following paper.

Sajal Halder, Kwan Hui Lim, Jeﬀrey Chan, and Xiuzhen Zhang. Deep Learning of Dynamic POI Generation and Optimisation for Itinerary Recommendation. Submitted to ACM Transactions on Recommender Systems, 2023. 

In this research work, we aim to answer the following research questions.

      (i) How does the proposed deep learning model select better next top-k POIs for the recommendation?  
     
      (ii) Is the proposed deep learning model effective for full itinerary recommendation?  
    
      (iii) How do temporal user interest, co-visiting patterns and personalisation features perform in the recommendation? 
    
      (iv) Why does greedy policy-based itinerary construction perform better than the Monte Carlo Tree Search (MCTS) based itinerary construction?
     
# Implemtation Details
 
In this DLIR model implemenation, we have used transformer and GCN that have been implemented in python programing language. We use tensorflow, keras and attention machanism.

Required Packages:

tensorflow: 2.4.1

pandas: 1.2.2

DLIR model has been implemented in mail.py file, model2.py contains some necessary files. 

Here we added only one dataset (Magic Kingdom). If you are interested to know about more datasets email at sajal.csedu01@gmail.com
