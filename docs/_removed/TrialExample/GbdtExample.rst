GBDT in nni
===========

Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion as other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.

Gradient boosting decision tree has many popular implementations, such as `lightgbm <https://github.com/Microsoft/LightGBM>`__\ , `xgboost <https://github.com/dmlc/xgboost>`__\ , and `catboost <https://github.com/catboost/catboost>`__\ , etc. GBDT is a great tool for solving the problem of traditional machine learning problem. Since GBDT is a robust algorithm, it could use in many domains. The better hyper-parameters for GBDT, the better performance you could achieve.

NNI is a great platform for tuning hyper-parameters, you could try various builtin search algorithm in nni and run multiple trials concurrently.

1. Search Space in GBDT
-----------------------

There are many hyper-parameters in GBDT, but what kind of parameters will affect the performance or speed? Based on some practical experience, some suggestion here(Take lightgbm as example):

..

   * For better accuracy
   * ``learning_rate``. The range of ``learning rate`` could be [0.001, 0.9].



* 
  ``num_leaves``. ``num_leaves`` is related to ``max_depth``\ , you don't have to tune both of them.

* 
  ``bagging_freq``. ``bagging_freq`` could be [1, 2, 4, 8, 10]

* 
  ``num_iterations``. May larger if underfitting.

..

   * For speed up
   * ``bagging_fraction``. The range of ``bagging_fraction`` could be [0.7, 1.0].



* 
  ``feature_fraction``. The range of ``feature_fraction`` could be [0.6, 1.0].

* 
  ``max_bin``.

..

   * To avoid overfitting
   * ``min_data_in_leaf``. This depends on your dataset.



* 
  ``min_sum_hessian_in_leaf``. This depend on your dataset.

* 
  ``lambda_l1`` and ``lambda_l2``.

* 
  ``min_gain_to_split``.

* 
  ``num_leaves``.

Reference link:
`lightgbm <https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html>`__ and `autoxgoboost <https://github.com/ja-thomas/autoxgboost/blob/master/poster_2018.pdf>`__

2. Task description
-------------------

Now we come back to our example "auto-gbdt" which run in lightgbm and nni. The data including :githublink:`train data <examples/trials/auto-gbdt/data/regression.train>` and :githublink:`test data <examples/trials/auto-gbdt/data/regression.train>`.
Given the features and label in train data, we train a GBDT regression model and use it to predict.

3. How to run in nni
--------------------

3.1 Install all the requirments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install lightgbm
   pip install pandas

3.2 Prepare your trial code
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You need to prepare a basic code as following:

.. code-block:: python

   ...

   def get_default_parameters():
       ...
       return params


   def load_data(train_path='./data/regression.train', test_path='./data/regression.test'):
       '''
       Load or create dataset
       '''
       ...

       return lgb_train, lgb_eval, X_test, y_test

   def run(lgb_train, lgb_eval, params, X_test, y_test):
       # train
       gbm = lgb.train(params,
                       lgb_train,
                       num_boost_round=20,
                       valid_sets=lgb_eval,
                       early_stopping_rounds=5)
       # predict
       y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

       # eval
       rmse = mean_squared_error(y_test, y_pred) ** 0.5
       print('The rmse of prediction is:', rmse)

   if __name__ == '__main__':
       lgb_train, lgb_eval, X_test, y_test = load_data()

       PARAMS = get_default_parameters()
       # train
       run(lgb_train, lgb_eval, PARAMS, X_test, y_test)

3.3 Prepare your search space.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you like to tune ``num_leaves``\ , ``learning_rate``\ , ``bagging_fraction`` and ``bagging_freq``\ , you could write a :githublink:`search_space.json <examples/trials/auto-gbdt/search_space.json>` as follow:

.. code-block:: json

   {
       "num_leaves":{"_type":"choice","_value":[31, 28, 24, 20]},
       "learning_rate":{"_type":"choice","_value":[0.01, 0.05, 0.1, 0.2]},
       "bagging_fraction":{"_type":"uniform","_value":[0.7, 1.0]},
       "bagging_freq":{"_type":"choice","_value":[1, 2, 4, 8, 10]}
   }

More support variable type you could reference `here <../Tutorial/SearchSpaceSpec.rst>`__.

3.4 Add SDK of nni into your code.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: diff

   +import nni
   ...

   def get_default_parameters():
       ...
       return params


   def load_data(train_path='./data/regression.train', test_path='./data/regression.test'):
       '''
       Load or create dataset
       '''
       ...

       return lgb_train, lgb_eval, X_test, y_test

   def run(lgb_train, lgb_eval, params, X_test, y_test):
       # train
       gbm = lgb.train(params,
                       lgb_train,
                       num_boost_round=20,
                       valid_sets=lgb_eval,
                       early_stopping_rounds=5)
       # predict
       y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

       # eval
       rmse = mean_squared_error(y_test, y_pred) ** 0.5
       print('The rmse of prediction is:', rmse)
   +   nni.report_final_result(rmse)

   if __name__ == '__main__':
       lgb_train, lgb_eval, X_test, y_test = load_data()
   +   RECEIVED_PARAMS = nni.get_next_parameter()
       PARAMS = get_default_parameters()
   +   PARAMS.update(RECEIVED_PARAMS)

       # train
       run(lgb_train, lgb_eval, PARAMS, X_test, y_test)

3.5 Write a config file and run it.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the config file, you could set some settings including:


* Experiment setting: ``trialConcurrency``\ , ``trialGpuNumber``\ , etc.
* Platform setting: ``trainingService``\ , etc.
* Path setting: ``searchSpaceFile``\ , ``trialCodeDirectory``\ , etc.
* Algorithm setting: select ``tuner`` algorithm, ``tuner optimize_mode``\ , etc.

An config.yml as follow:

.. code-block:: yaml

   experimentName: auto-gbdt example
   searchSpaceFile: search_space.json
   trialCommand: python3 main.py
   trialGpuNumber: 0
   trialConcurrency: 1
   maxTrialNumber: 10
   trainingService:
     platform: local
   tuner:
     name: TPE  #choice: TPE, Random, Anneal, Evolution, BatchTuner, etc
     classArgs:
       optimize_mode: minimize

Run this experiment with command as follow:

.. code-block:: bash

   nnictl create --config ./config.yml
