import logging
import os
import sys
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/")
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/utils/")
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_analysis_weights/")
import ml_utils as ml_utils
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config.conf import setup_logging
import random


def split_stage_4(FILE_PATH_TRAIN, FILE_PATH_TEST, lq_mass_train, lq_mass_test):
    # 0: for no lq_all
    # 1: for all the masses train except for tested one
    # 2: for all the masses, tested one included, just 20%  
    ratio_2 = 0
    lq_all_approach = 2
    data_reduction = 0
    reduction_percentage = 100
    random_state = 42
    random_state_2 = 42
    train_split = 80
    test_split = 100 - train_split
    FILE_PATH_TRAIN = FILE_PATH_TRAIN + "/"
    FILE_PATH_TEST = FILE_PATH_TEST + "/"
    
   

    setup_logging("SplittingLogger", file_handler=False)
    logger = logging.getLogger("SplittingLogger")

    # Two mass approaches
    if lq_mass_test != lq_mass_train and (lq_all_approach == 1 or lq_all_approach == 0): 

        X_tr = ml_utils.load(FILE_PATH_TRAIN + 'X.pkl')
        y_tr = ml_utils.load(FILE_PATH_TRAIN + 'y.pkl')
        f_tr = ml_utils.load(FILE_PATH_TRAIN + 'f.pkl')
        w_tr = ml_utils.load(FILE_PATH_TRAIN + 'w.pkl')
        y_lq_all_tr = []
        if lq_all_approach == 1:#lq_mass_train == 'lq_all':
            print("LQ_ALL used")
            y_lq_all_tr = ml_utils.load(FILE_PATH_TRAIN + 'y_lq_all.pkl')
        else:
            print("Different masses")
        X_te = ml_utils.load(FILE_PATH_TEST + 'X.pkl')
        y_te = ml_utils.load(FILE_PATH_TEST + 'y.pkl')
        f_te = ml_utils.load(FILE_PATH_TEST + 'f.pkl')
        w_te = ml_utils.load(FILE_PATH_TEST + 'w.pkl')

        if data_reduction == 1:
            X_tr, X_test_1, y_tr, y_test_1, w_tr, w_test_1 = train_test_split(X_tr, y_tr, w_tr, random_state=random.randint(0, 1000), train_size=reduction_percentage / 100)
            X_te, X_test_2, y_te, y_test_2, w_te, w_test_2 = train_test_split(X_te, y_te, w_te, random_state=random.randint(0, 1000), train_size=reduction_percentage / 100)


        # Shuffle and split data

        X_train_1, X_test_1, y_train_1, y_test_1, w_train_1, w_test_1 = train_test_split(X_tr, y_tr, w_tr, random_state=random.randint(0, 1000), train_size=train_split / 100)
        X_train_2, X_test_2, y_train_2, y_test_2, w_train_2, w_test_2 = train_test_split(X_te, y_te, w_te, random_state=random.randint(0, 1000), train_size=train_split / 100)

        # Splits data and returns correct ratio after negative weights removal
        tmp = 0

        df_train_1 = pd.DataFrame(data=X_train_1)
        df_train_1['weights'] = w_train_1
        df_train_1['y'] = y_train_1
        df_train_1['is_selected'] = df_train_1.weights > 0
        df_train_1 = df_train_1[df_train_1.is_selected]

        df_train_2 = pd.DataFrame(data=X_train_2)
        df_train_2['weights'] = w_train_2
        df_train_2['y'] = y_train_2
        df_train_2['is_selected'] = df_train_2.weights > 0
        df_train_2 = df_train_2[df_train_2.is_selected]

        # remove LQ from the particural mass df and create sub df for later concat
        df_test_1 = pd.DataFrame(data=X_test_1)
        df_test_1['weights'] = w_test_1
        df_test_1['y'] = y_test_1
        df_test_1['is_lq'] = df_test_1.y == 0
        df_test_1['not_lq'] = df_test_1.y != 0
        df_test_lq_1 = df_test_1[df_test_1.is_lq] # just LQ
        df_test_1 = df_test_1[df_test_1.not_lq] # without LQ

        df_test_2 = pd.DataFrame(data=X_test_2)
        df_test_2['weights'] = w_test_2
        df_test_2['y'] = y_test_2
        df_test_2['is_lq'] = df_test_2.y == 0
        df_test_2['not_lq'] = df_test_2.y != 0
        df_test_lq_2 = df_test_2[df_test_2.is_lq]
        df_test_2 = df_test_2[df_test_2.not_lq]


        # concat bg and signal for the testing parts
        # cross concat to achieve bg from the first mass but LQ from the second
        # now we obtained shuffeled testing part

        #following is just for the comparison purpose
        #-----------------------------------------------------------------
        df_test_1_tmp = pd.concat([df_test_1, df_test_lq_1])
        df_test_2_tmp = pd.concat([df_test_2, df_test_lq_2])
        
        df_1_tmp = pd.concat([df_train_1, df_test_1_tmp])
        df_2_tmp  = pd.concat([df_train_2, df_test_2_tmp])
    
        #-----------------------------------------------------------------




    
        print("New ratio has to be calculated.")
        df_test_1 = pd.concat([df_test_1, df_test_lq_2])
        df_test_2 = pd.concat([df_test_2, df_test_lq_1])




        # for train parts
        w_train_1 = df_train_1.weights.to_numpy()
        y_train_1 = df_train_1.y.to_numpy()
        df_train_1 = df_train_1.drop(columns= ["weights", "y", "is_selected"])
        X_train_1 = df_train_1.to_numpy()
        

        w_train_2 = df_train_2.weights.to_numpy()
        y_train_2 = df_train_2.y.to_numpy()
        df_train_2 = df_train_2.drop(columns= ["weights", "y", "is_selected"])
        X_train_2 = df_train_2.to_numpy()

        # for test parts
        w_test_1 = df_test_1.weights.to_numpy()
        y_test_1 = df_test_1.y.to_numpy()
        df_test_1 = df_test_1.drop(columns= ["weights", "y", "is_lq", "not_lq"])
        X_test_1 = df_test_1.to_numpy()
        

        w_test_2 = df_test_2.weights.to_numpy()
        y_test_2 = df_test_2.y.to_numpy()
        df_test_2 = df_test_2.drop(columns= ["weights", "y", "is_lq", "not_lq"])
        X_test_2 = df_test_2.to_numpy()
        
        df_1 = pd.concat([df_train_1, df_test_1])


        ratio_1 = float(len(df_train_1))/float(len(df_1_tmp))
        print('Old ratio : '+ str(ratio_1))
        ratio_2 = float(len(df_train_1))/float(len(df_1))
        print('New ratio after lq swap: '+ str(ratio_2))

        # X_train_1, y_train_1, w_train_1, f_tr = ml_utils.check_feature_order(X_train_1,
        #  y_train_1, w_train_1, f_tr)

        # X_test_1, y_test_1, w_test_1, f_tr = ml_utils.check_feature_order(X_test_1,
        #  y_test_1, w_test_1, f_tr)

        # X_train_2, y_train_2, w_train_2, f_te = ml_utils.check_feature_order(X_train_2,
        #  y_train_2, w_train_2, f_te)

        # X_test_2, y_test_2, w_test_2, f_tr = ml_utils.check_feature_order(X_test_2,
        #  y_test_2, w_test_2, f_te)

        #/train_mass/test_mass/
        ml_utils.save(X_train_1, FILE_PATH_TRAIN  + lq_mass_test, 'X_train')
        ml_utils.save(y_train_1, FILE_PATH_TRAIN  + lq_mass_test, 'y_train')
        ml_utils.save(w_train_1, FILE_PATH_TRAIN  + lq_mass_test, 'w_train')
        ml_utils.save(X_test_1, FILE_PATH_TRAIN  + lq_mass_test, 'X_test')
        ml_utils.save(y_test_1, FILE_PATH_TRAIN  + lq_mass_test, 'y_test')
        ml_utils.save(w_test_1, FILE_PATH_TRAIN + lq_mass_test, 'w_test')
        ml_utils.save(f_tr, FILE_PATH_TRAIN  + lq_mass_test, 'f_new')

        #/test_mass/train_mass/
        ml_utils.save(X_train_2, FILE_PATH_TEST  + lq_mass_train, 'X_train')
        ml_utils.save(y_train_2, FILE_PATH_TEST + lq_mass_train, 'y_train')
        ml_utils.save(w_train_2, FILE_PATH_TEST + lq_mass_train, 'w_train')
        ml_utils.save(X_test_2, FILE_PATH_TEST + lq_mass_train, 'X_test')
        ml_utils.save(y_test_2, FILE_PATH_TEST + lq_mass_train, 'y_test')
        ml_utils.save(w_test_2, FILE_PATH_TEST + lq_mass_train, 'w_test')
        ml_utils.save(f_te, FILE_PATH_TEST + lq_mass_train, 'f_new')

        return 0.8

    # one mass approaches
    if lq_mass_train == lq_mass_test or lq_all_approach == 2:
        logger.info("Just one mass file was used")
        X = ml_utils.load(FILE_PATH_TRAIN + 'X.pkl')
        y = ml_utils.load(FILE_PATH_TRAIN + 'y.pkl')
        f = ml_utils.load(FILE_PATH_TRAIN + 'f.pkl')
        w = ml_utils.load(FILE_PATH_TRAIN + 'w.pkl')
        y_lq_all = None
        y_lq_all_test = None

        if lq_all_approach == 2:
            if lq_all_approach == 2:
                logger.info("LQ_ALL used")
                y_lq_all = ml_utils.load(FILE_PATH_TRAIN + 'y_lq_all.pkl')
            else:
                logger.info("Without LQ_ALL used")
        print("------------------------------------------------------------------")
        df_tmp = pd.DataFrame(data=X)
        df_tmp['y_lq_all'] = y_lq_all
        df_tmp['is_lq'] = df_tmp.y_lq_all == float(lq_mass_test.split('_',1)[1]) #((df_tmp.y_lq_all >= 500.0) & (df_tmp.y_lq_all <= 1600.0))
        df_tmp = df_tmp[df_tmp.is_lq]
        all = len(df_tmp)
        print("------------------------------------------------------------------")
           
        if lq_all_approach == 2:
            X_train, X_test, y_train, y_test, w_train, w_test, y_lq_all_train, y_lq_all_test = train_test_split(X, y, w, y_lq_all,random_state=random.randint(0, 1000), train_size=train_split / 100)
        else:
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, random_state=random.randint(0, 1000), train_size=train_split / 100)

        print('X_MATRIX: '+ str(len(X_train)))

        #for the training part
        df_train = pd.DataFrame(data=X_train)
        df_train['weights'] = w_train

        df_train['y'] = y_train
        if lq_all_approach == 2:
            df_train['y_lq_all_train'] = y_lq_all_train
    #    df_train['y_lq_all'] = y_lq_all_train
        df_train['is_selected'] = df_train.weights > 0
        df_train = df_train[df_train.is_selected]
        
        

        #for the testing part
        if lq_all_approach == 2:
            df_test = pd.DataFrame(data=X_test)
            df_test['weights'] = w_test
            df_test['y'] = y_test
            df_test['y_lq_all_test'] = y_lq_all_test
            mask = float(lq_mass_test.split('_',1)[1])
            # print(mask)
            df_test['test_rem'] =  (df_test.y_lq_all_test == mask) | (df_test.y > 0.1) #| \
                            #       (df_test.y > 0.1)
            df_test = df_test[df_test.test_rem]
            print("lenghth of the df: "+ str(len(df_test)))
            
         #--------------------CHECKER----------------------

            # counter = [0,0,0,0,0,0,0,0,0,0,0,0]
            # for i ,event in df_train.iterrows():
            #     if event.y_lq_all == 500.0:
            #         counter[0] += 1
            #     elif event.y_lq_all == 600.0:
            #         counter[1] += 1
            #     elif event.y_lq_all == 700.0:
            #         counter[2] += 1
            #     elif event.y_lq_all == 800.0:
            #         counter[3] += 1
            #     elif event.y_lq_all == 900.0:
            #         counter[4] += 1
            #     elif event.y_lq_all == 1000.0:
            #         counter[5] += 1
            #     elif event.y_lq_all == 1100.0:
            #         counter[6] += 1
            #     elif event.y_lq_all == 1200.0:
            #         counter[7] += 1
            #     elif event.y_lq_all == 1300.0:
            #         counter[8] += 1
            #     elif event.y_lq_all == 1400.0:
            #         counter[9] += 1
            #     elif event.y_lq_all == 1500.0:
            #         counter[10] += 1
            #     elif event.y_lq_all == 1600.0:
            #         counter[11] += 1
            # counter2 = [0]
            # for i ,event in df_test.iterrows():
            #     if event.y_lq_all == 500.0:
            #         counter2[0] += 1

            # print("Number of events without other lq: "+str(counter))
            # print("Number of events without other lq: "+str(counter2))

            # counter = [0,0,0,0,0,0,0,0,0,0,0,0]
            # for i ,event in df_train.iterrows():
            #     if (event.y_lq_all == float(lq_mass_test.split('_',1)[1])) | (event.y != 0.0):
            #         print(str(event.y) +' '+ str(event.y_lq_all))
            #         counter += 1

            # print("Number of events without other lq: "+str(counter))
            
        #-----------------------END------------------------

            #ml_utils.save(X_train, FILE_PATH_TRAIN + lq_mass_test, 'X_train')
            y_lq_all_test = df_test['y_lq_all_test']
            #df_test = df_test.drop(columns=["y_lq_all_test"])
            
            print("lq_all_test_len: "+ str(len(df_test)))
        else:
            df_test = pd.DataFrame(data=X_test)
            df_test['weights'] = w_test
            df_test['y'] = y_test



        #--------------------CHECKER----------------------
        # print(len(X_all))
        # print(len(X))


        # counter = 0
        # for i ,event in df_train.iterrows():
        #     if event.weights < 0:
        #         print(str(event.y) +' '+ str(event.weights))
        #         counter += 1

        # print("Number of events with negative value: "+str(counter))
            
        #-----------------------END------------------------
        w_train = df_train.weights.to_numpy()
        y_train = df_train.y.to_numpy()
        if lq_all_approach == 2:
            y_lq_all_train = df_train.y_lq_all_train.to_numpy()
        df_train = df_train.drop(columns= ["weights", "y", "is_selected"])
        if lq_all_approach == 2:
            df_train = df_train.drop(columns=[ "y_lq_all_train"])
        X_train = df_train.to_numpy()


        w_test = df_test.weights.to_numpy()
        y_test = df_test.y.to_numpy()
        if lq_all_approach == 2:
            y_lq_all_test = df_test.y_lq_all_test.to_numpy()
        if lq_all_approach == 2:
            df_test = df_test.drop(columns= ["weights", "y", 'test_rem', 'y_lq_all_test'])
        else:    
            df_test = df_test.drop(columns= ["weights", "y"])

        X_test = df_test.to_numpy()

        df_tmp = pd.DataFrame(data=X_test)
        if lq_all_approach == 2:
            df_tmp['y_lq_all_test'] = y_lq_all_test
            df_tmp['is_lq'] = df_tmp.y_lq_all_test == float(lq_mass_test.split('_',1)[1])  # ((df_tmp.y_lq_all >= 500.0) & (df_tmp.y_lq_all <= 1600.0))
            df_tmp = df_tmp[df_tmp.is_lq]
            have = len(df_tmp)

        if data_reduction == 1:
            if lq_all_approach == 2:
                X_train, X_t, y_train, y_t, w_train, w_t = train_test_split(X_train, y_train, w_train, random_state=random_state_2, train_size=reduction_percentage / 100)
                X_test, X_t, y_test, y_t, w_test, w_t = train_test_split(X_test, y_test, w_test, random_state=random_state_2, train_size=reduction_percentage / 100)
                

            else:    
                X, X_test, y, y_test, w, w_test = train_test_split(X, y, w, random_state=random_state, train_size=reduction_percentage / 100)




        ml_utils.save(X_train, FILE_PATH_TRAIN  + lq_mass_test, 'X_train')
        ml_utils.save(y_train, FILE_PATH_TRAIN  + lq_mass_test, 'y_train')
        ml_utils.save(w_train, FILE_PATH_TRAIN  + lq_mass_test, 'w_train')
        if lq_all_approach == 2:
            ml_utils.save(y_lq_all_train, FILE_PATH_TRAIN + lq_mass_test, 'y_lq_all_train')
        ml_utils.save(X_test, FILE_PATH_TRAIN  + lq_mass_test, 'X_test')
        ml_utils.save(y_test, FILE_PATH_TRAIN  + lq_mass_test, 'y_test')
        ml_utils.save(w_test, FILE_PATH_TRAIN  + lq_mass_test, 'w_test')
        ml_utils.save(y_lq_all_test, FILE_PATH_TRAIN + lq_mass_test, 'y_lq_all_test')
        ml_utils.save(f, FILE_PATH_TRAIN  + lq_mass_test, 'f_new')
        if lq_all_approach == 2:
            ratio_2 = (have/all)*(reduction_percentage/100)

        else:
            ratio_2 = 0.2
        return ratio_2
    return ratio_2
        
        




