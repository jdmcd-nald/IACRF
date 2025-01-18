import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from skgarden import RandomForestQuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import pickle
import time
import os
from pprint import pprint

def error_evalue(y_hat, y):
    return np.sum(np.abs(y_hat - y)) / len(y)


def normal_x(x):
    # print(x.shape)
    # name_list = ['Al', 'Zn', 'Mg', 'Cu', 'Si', 'Fe', 'Mn', 'Cr', 'Ti', 'Other', 'Cycle',
    #              'Temper', 'Precip', 'pH', 'Cl', 'wf']
    name_list = ['Cyc', 'Al', 'Zn', ]
    variable_N = x.shape[1]  # 15
    sample_N = x.shape[0]  # 150
    data = {}
    for i in range(variable_N):
        data[name_list[i]] = []
        for j in range(sample_N):
            data[name_list[i]].append(x[j][i])
    # print(data)
    # Find the Max data and Min data
    for h in name_list:
        h_max = max(data[h])
        h_min = min(data[h])
        length = "{:.5f}".format(h_max - h_min)
        # print(h, h_max, h_min, length)

        # Normalization
        h_list = []
        for h_data in data[h]:
            normal_data = (float(h_data) - h_min) / float(length)
            h_list.append(round(normal_data, 5))
        # print(h_list)

        data[h] = h_list

    # consist array
    normal_list = []
    for i in range(sample_N):
        data2 = []
        for j in range(variable_N):
            data2.append(data[name_list[j]][i])
        normal_list.append(data2)

    # print(normal_list)
    return np.array(normal_list)


def normal_y(y):
    sample_N = y.shape[0]
    y_max = max(y)
    y_min = min(y)
    length = float(y_max - y_min)
    new_list = []
    for j in range(sample_N):
        y_new = round((y[j] - y_min) / length, 5)
        new_list.append(y_new)

    return new_list


if __name__ == '__main__':

    file_vector = ['data/Training.csv',
                   'data/Training_2.csv',
                   'data/Training_3.csv',
                   'data/Training_4.csv',
                   'data/Training_5.csv']
    
    file_count = 1
    for file_names in file_vector:
        name = 'varied_results'
        outdir = 'results/'
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        df = pd.read_csv(file_names)
        a = np.array(df)
        YY = a[:, -1]
        X = a[:, :-1]
        # print(X[0], Y[0])
        # exit()
        x_normal = X  # normal_x(X)
        Y_vec = [YY,np.log(YY)]
        Y_count=1
        for Y in Y_vec :
            y_normal = Y  # np.log(Y) #  # normal_y(Y)
            count = 0
            for i in y_normal:
                if (i == 0):
                    y_normal[count] = .00001
                count = count + 1
            print(y_normal)
            #ts = pd.read_csv("/Users/user/Documents/Python Projects/Womanium/ICARF/data/Valid.csv")
            #tsa = np.array(ts)
            #ts_x = tsa[:, :-1]
            #ts_y = tsa[:, -1]


            variants = {'criterion': ['absolute_error','squared_error', 'friedman_mse'],
                   'max_depth': [10, 20, 40, 50, 80, 200, None],
                   'min_impurity_decrease': [0.0, 0.2, 0.5,1, 2,4],
                   'max_features': [None, 'log2', 'sqrt'],
                   'max_leaf_nodes': [None,2,3,5, 7,10],
                   'min_samples_leaf': [1, 2, 3,4, 10],
                   'min_samples_split': [2, 3,4,5, 7,10],
                   'min_weight_fraction_leaf': [0.0, .02,0.05, 0.175,0.3, 0.35, 0.5],
                   'n_estimators': [35,50, 75, 100,200],
                   'warm_start' : [False],
                   'bootstrap': [True],
                   'n_jobs': [-1],
                   'oob_score': [True]
                   }
            best_score_parameters = variants
            best_accuracy_parameters = variants
            # x_train, x_test, y_train, y_test = train_test_split(x_normal, y_normal, test_size=0.3, random_state=None)
            #record = []
            #iii = 0
            #iii_mean = 0
            #iii_list = []
            #iii_std = 0
            #iii_tot = 0
            #iii_tot_mean = 0
            #iii_tot_list = []
            #iii_tot_std = 0
            current_best = 0
            current_best_accuracy = 100000
            rf = RandomForestRegressor()

            rs = ShuffleSplit(n_splits=30, test_size=0.3)
            # dont think I can use random_state here or it collapses possibillities
            rf_random = RandomizedSearchCV(estimator=rf, param_distributions=variants , n_iter=20  , cv=3, verbose=0, n_jobs=-1)
            # Fit the random search model

            for train_idx, test_idx in rs.split(x_normal):
                rf_random.fit(x_normal[train_idx], y_normal[train_idx])
                # score = rf_random.score(x_normal[test_idx], y_normal[test_idx])
                score = rf_random.best_score_
                example = y_normal[test_idx]
                predictions = rf_random.predict(x_normal[test_idx])
                errors = np.abs(predictions - y_normal[test_idx])
                new_errors = np.abs(errors / y_normal[test_idx])

                normalized_error = np.divide(np.abs(errors),
                                             np.minimum(np.abs(predictions), np.abs(y_normal[test_idx])))

                mape = 100 * np.mean(new_errors)
                accuracy = 100 - mape

                if (score > current_best):
                    print(file_count, "+", Y_count, ' Model Performance')
                    print('Average Error: {:0.4f} .'.format(np.mean(errors)))
                    print('Accuracy = {:0.2f}%.'.format(accuracy))
                    print('MAPE = {:0.2f}%.'.format(mape))
                    current_best = score
                    print("score :", score)
                    print("norm errors :", normalized_error)
                    print(rf_random.best_params_)
                    best_score_parameters = rf_random.best_params_

                if (abs(mape) < current_best_accuracy):
                    print(file_count, "+", Y_count, ' Accuracy Model Performance')
                    print('Average Error: {:0.4f} .'.format(np.mean(errors)))
                    print('Accuracy = {:0.2f}%.'.format(accuracy))
                    print('MAPE = {:0.2f}%.'.format(mape))
                    current_best_accuracy = abs(mape)
                    print("score :", score)
                    print("norm errors :", normalized_error)
                    print(rf_random.best_params_)
                    best_accuracy_parameters = rf_random.best_params_

            parameter_list = [best_accuracy_parameters,best_score_parameters]
            parameter_list.append({'n_estimators':100,
                                            'max_depth':None,
                                            'max_features':None,
                                            'min_samples_split':2,
                                            'min_samples_leaf':1,
                                            'min_weight_fraction_leaf':0.0,
                                            'max_leaf_nodes':None,
                                            'bootstrap':True,
                                            'oob_score':True,
                                            'n_jobs':-1,
                                            'verbose':0,
                                            'warm_start':False})
            parameter_list.append({'n_estimators': 100,
                                   'max_depth': 5,
                                   'max_features': 'sqrt',
                                   'min_samples_split': 3,
                                   'min_samples_leaf': 2,
                                   'min_weight_fraction_leaf': 0.02,
                                   'max_leaf_nodes': 30,
                                   'bootstrap': True,
                                   'oob_score': True,
                                   'n_jobs': -1,
                                   'verbose': 0,
                                   'warm_start': False})

            for parameter in parameter_list:
                record = []
                iii = 0
                iii_mean = 0
                iii_list = []
                iii_std = 0
                iii_tot = 0
                iii_tot_mean = 0
                iii_tot_list = []
                iii_tot_std = 0
                current_best = 0
                current_best_accuracy = 1000
                rf = RandomForestRegressor(parameter)
                rs = ShuffleSplit(n_splits=300, test_size=0.3)
                j = 0

                for train_idx, test_idx in rs.split(x_normal):

                    rf_random.fit(x_normal[train_idx], y_normal[train_idx])
                    score = rf_random.score(x_normal[test_idx], y_normal[test_idx])
                    example = y_normal[test_idx]
                    predictions = rf_random.predict(x_normal[test_idx])
                    errors = np.abs(predictions - y_normal[test_idx])
                    new_errors = np.abs(errors / y_normal[test_idx])
                    mape = 100 * np.mean(new_errors)
                    accuracy = 100 - mape

                    normalized_error = np.divide(np.abs(errors),np.minimum(np.abs(predictions),np.abs(y_normal[test_idx])))

                    iii_tot = iii_tot + 1
                    iii_tot_mean = iii_tot_mean + score
                    iii_tot_list.append(score)

                    if (score > current_best):
                        record.append((str(file_count)+" + "+str(Y_count)+' (Accu) Model Performance'))
                        record.append('Average Error: {:0.4f} .'.format(np.mean(errors)))
                        record.append('Accuracy = {:0.2f}%.'.format(accuracy))
                        record.append('MAPE = {:0.2f}%.'.format(mape))
                        record.append("score :"+str(score))
                        record.append("norm errors :"+str(normalized_error))
                        record.append("\n")
                        print(file_count, "+", Y_count, ' (Accu) Model Performance')
                        print('Average Error: {:0.4f} .'.format(np.mean(errors)))
                        print('Accuracy = {:0.2f}%.'.format(accuracy))
                        print('MAPE = {:0.2f}%.'.format(mape))
                        current_best = score
                        print("score :", score)
                        print("norm errors :", normalized_error)

                    if (abs(mape) < current_best_accuracy):
                        record.append(str(file_count)+"+"+str(Y_count)+' (Accu) Accuracy Model Performance')
                        record.append('Average Error: {:0.4f} .'.format(np.mean(errors)))
                        record.append('Accuracy = {:0.2f}%.'.format(accuracy))
                        record.append('MAPE = {:0.2f}%.'.format(mape))
                        record.append("score :"+str(score))
                        record.append("norm errors :"+str(normalized_error))
                        record.append("\n")
                        print(file_count, "+", Y_count, ' (Accu) Accuracy Model Performance')
                        print('Average Error: {:0.4f} .'.format(np.mean(errors)))
                        print('Accuracy = {:0.2f}%.'.format(accuracy))
                        print('MAPE = {:0.2f}%.'.format(mape))
                        current_best_accuracy = abs(mape)
                        print("score :", score)
                        print("norm errors :", normalized_error)

                    if score > 0.89:  # and score2 > 0.5:
                        tm = time.strftime("%Y%m%d-%H%M%S", time.localtime())
                        score_100 = round(score, 2) * 100
                        # score2_100 = round(score2, 2) * 100
                        record_name = '%d-%d-%d' % (score_100, 1, j)

                        record_per = '%d    %d  %.5f   ' % (1, j, score)
                        print(record_per)
                        record.append(record_per)
                        iii = iii + 1
                        iii_mean = iii_mean + score
                        iii_list.append(score)

                if (iii > 0):
                    iii_mean = iii_mean / iii
                    for elem in iii_list:
                        iii_std = iii_std + (elem - iii_mean) ** 2
                    iii_std = np.sqrt(iii_std / iii)
                iii_tot_mean = iii_tot_mean / iii_tot
                for elem in iii_tot_list:
                    iii_tot_std = iii_tot_std + (elem - iii_tot_mean) ** 2
                iii_tot_std = np.sqrt(iii_tot_std / iii_tot)
                record.append(str(file_count)+ "+" + str(Y_count)+" results:")
                record.append(str(iii)+" + "+str(iii_mean)+" + "+str(iii_std))
                record.append(str(iii_tot)+" + "+str(iii_tot_mean)+" + "+str(iii_tot_std))
                record.append(str(max(iii_tot_list)))
                print(file_count, "+", Y_count, " results:")
                print(iii, iii_mean, iii_std)
                print(iii_tot, iii_tot_mean, iii_tot_std)
                print(max(iii_tot_list))
                Y_count = Y_count + 1
                with open('results/%s.txt' % (name), 'a+') as f:
                    f.write('\n'.join(record))
        file_count = file_count+1


