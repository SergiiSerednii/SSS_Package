__version__ = '2020.05.26'
__Author__ = 'Serednii Sergii'
__contacts__ = ['sseredniy@gmail.com', 'linkedin.com/in/sergii-serednii-ab433815']

################################################### Import packages ####################################################
import pandas as pd
from pandas import read_csv as p_read_csv, to_datetime, to_numeric, qcut
import numpy as np
from numpy import log
from sklearn.tree import DecisionTreeClassifier
import os
import warnings
from xgboost import rabit
from xgboost import callback
from xgboost.training import _train_internal
from xgboost.core import EarlyStopException
from xgboost.callback import _fmt_metric
import time
from tqdm import tqdm
from hyperopt import base
from hyperopt.fmin import generate_trials_to_calculate, space_eval
from hyperopt.utils import coarse_utcnow
from hyperopt.std_out_err_redirect_tqdm import std_out_err_redirect_tqdm
import logging
from psycopg2 import connect
from psycopg2.extras import (DictCursor, )
from psycopg2.extras import execute_values
import pickle
import inspect
import re
import types

################################################## MyModule functions ##################################################
# Console colors
CS = '\33[40m' + '\33[37m' # '\33[43m' + '\33[34m'
CE = '\033[0m'

def read_csv(filename, sep = ',', str_as_category=True, date_as_date=True):
    data = p_read_csv(filename, sep)

    for i in data.columns[data.dtypes == 'O']:
        if str_as_category:
            data[i] = data[i].astype('category')
        if date_as_date:
            try:
                data[i] = to_datetime(data[i])
            except:
                pass

    return data

def WoE_full(df, var, target):
    # Calculate number of goods and bads
    rez = df.groupby(var).agg({target: {'Goods': lambda x: x.shape[0] - sum(x), 'Bads': 'sum'}})

    # Avoid nan categories
    rez.fillna(0, inplace=True)

    # Avoid division by zero by adding 1 to numerator and denominator
    mask = (rez[target]['Goods'] == 0) | (rez[target]['Bads'] == 0)
    rez[mask] = rez[mask] + 1

    # Calculate WoE
    rez['GR'] = rez[target]['Goods'] / rez[target]['Goods'].sum()
    rez['BR'] = rez[target]['Bads'] / rez[target]['Bads'].sum()
    rez['WoE'] = log(rez['GR']/rez['BR'])

    return rez

def WoE(df, var, target):
    rez = WoE_full(df, var, target)

    return rez['WoE']

def IV(df, var, target, n_levels_to_factor_threshold = 5, calc_type = 'Categorical', Min_Category_Share = 0.05, nbins = 10):
    # cut numeric features
    if is_numeric(df[var]) and len(df[var].unique()) > n_levels_to_factor_threshold:
        if calc_type == 'Interval':
            df = pd.DataFrame({var: qcut(df[var], q=nbins, duplicates='drop'), target: df[target]})

        elif calc_type == 'Categorical':
            tree = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=int(df.shape[0]*Min_Category_Share), presort=True, random_state=1223)
            tree.fit(df.loc[~df[var].isna(), [var]], df.loc[~df[var].isna(), target])

            tmp = tree.apply(df.loc[~df[var].isna(), [var]]).astype(str)
            tmp = ['leaf_' + e for e in tmp]

            tmp2 = ~df[var].isna()

            df = pd.DataFrame({var: ['NA']*df.shape[0], target: df[target]})
            df.loc[tmp2, var] = tmp

    # calculate IV
    rez = WoE_full(df, var, target)
    rez = sum(rez['WoE'] * (rez['GR'] - rez['BR']))

    # return result
    return rez

def PSI(df, var, sample_bin, nbins = 10, n_levels_to_factor_threshold = 5):
    # check that var isn't all nulls column
    if df[var].isna().all():
        return None

    # Save original value (in case of cut)
    sc = df[var]
    # cut numeric features
    if is_numeric(df[var]) and len(df[var].unique()) > n_levels_to_factor_threshold:
        if df[var].dtype == 'O':
            df[var] = to_numeric(df[var])
        df[var] = qcut(df[var], q=nbins, duplicates='drop')

    # calculate aggregates of categories
    tmp = df.groupby(var).agg({sample_bin: {'OOT': lambda x: sum(x == 'OOT'), 'Main': lambda x: sum(x == 'Main')}})

    # Avoid nan categories
    tmp.fillna(0, inplace=True)

    # Avoid division by zero by adding 1 to numerator and denominator
    mask = (tmp[sample_bin]['Main'] == 0) | (tmp[sample_bin]['OOT'] == 0)
    tmp[mask] = tmp[mask] + 1

    # Calculate ratios
    tmp[sample_bin] = tmp[sample_bin].astype('float')
    tmp[sample_bin]['Main'] /= tmp[sample_bin]['Main'].sum()
    tmp[sample_bin]['OOT'] /= tmp[sample_bin]['OOT'].sum()

    # return original value
    df[var] = sc

    # calculate PSI
    rez = (tmp[sample_bin]['OOT'] - tmp[sample_bin]['Main']) * (log(tmp[sample_bin]['OOT'] / tmp[sample_bin]['Main']))
    rez = sum(rez)

    # return result
    return rez

def is_numeric(series):
    if series.dtype.name == 'category':
        series = series.astype('str')

    return to_numeric(series.fillna(0), errors='coerce').notnull().all()

def listdiff(list1, list2):
    list2 = set(list2)
    rez = [x for x in list1 if x not in list2]

    return rez

def list_intersect(list1, list2):
    list2 = set(list2)
    rez = [x for x in list1 if x in list2]

    return rez

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

def draw_tree(tree, path):
    dot_data = StringIO()

    dot_data.seek(0)  # to reset graph?
    dot_data.truncate(0)  # to reset graph?

    export_graphviz(tree, out_file=dot_data, feature_names=tree.feature_names, filled=True, impurity=False, proportion=True, rounded=True, special_characters=True, node_ids=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    open(path, 'wb').write(Image(graph.create_png()).data)

#################################################### Block Formater ####################################################
def block_formater(block_name = '', filler = '#'):
    i = (118 - len(block_name)) // 2
    rez = ''.join(np.repeat(filler, i)) + ' ' + block_name + ' ' + ''.join(np.repeat(filler, 118 - len(block_name) - i))
    _ = os.system("echo %s | clip" % rez)
    return rez

############################################## Modified XGBoost functions ##############################################
def early_stop_s(stopping_rounds, max_overfit, symetric_overfit=False, maximize=False, verbose=True):
    """Create a callback that activates early stoppping.
    Validation error needs to decrease at least
    every **stopping_rounds** round(s) to continue training.
    Requires at least one item in **evals**.
    If there's more than one, will use the last.
    Returns the model from the last iteration (not the best one).
    If early stopping occurs, the model will have three additional fields:
    ``bst.best_score``, ``bst.best_iteration`` and ``bst.best_ntree_limit``.
    (Use ``bst.best_ntree_limit`` to get the correct value if ``num_parallel_tree``
    and/or ``num_class`` appears in the parameters)
    Parameters
    ----------
    stopp_rounds : int
       The stopping rounds before the trend occur.
    maximize : bool
        Whether to maximize evaluation metric.
    verbose : optional, bool
        Whether to print message about early stopping information.
    Returns
    -------
    callback : function
        The requested callback function.
    """
    state = {}

    def init(env):
        """internal function"""
        bst = env.model

        if not env.evaluation_result_list:
            raise ValueError('For early stopping you need at least one set in evals.')
        if len(env.evaluation_result_list) > 1 and verbose:
            msg = ("Multiple eval metrics have been passed: "
                   "'{0}' will be used for early stopping.\n\n")
            rabit.tracker_print(msg.format(env.evaluation_result_list[-1][0]))
        maximize_metrics = ('auc', 'aucpr', 'map', 'ndcg')
        maximize_at_n_metrics = ('auc@', 'aucpr@', 'map@', 'ndcg@')
        maximize_score = maximize
        metric_label = env.evaluation_result_list[-1][0]
        metric = metric_label.split('-', 1)[-1]

        if any(metric.startswith(x) for x in maximize_at_n_metrics):
            maximize_score = True

        if any(metric.split(":")[0] == x for x in maximize_metrics):
            maximize_score = True

        if verbose and env.rank == 0:
            msg = "Will train until {} hasn't improved in {} rounds.\n"
            rabit.tracker_print(msg.format(metric_label, stopping_rounds))

        state['maximize_score'] = maximize_score
        state['best_iteration'] = 0
        state['best_msg'] = ''
        if maximize_score:
            state['best_score'] = float('-inf')
            state['best_score_train'] = float('-inf')
        else:
            state['best_score'] = float('inf')
            state['best_score_train'] = float('inf')

        if bst is not None:
            if bst.attr('best_score') is not None:
                state['best_score'] = float(bst.attr('best_score'))
                state['best_iteration'] = int(bst.attr('best_iteration'))
                state['best_msg'] = bst.attr('best_msg')
            else:
                bst.set_attr(best_iteration=str(state['best_iteration']))
                bst.set_attr(best_score=str(state['best_score']))
                bst.set_attr(best_score_train=str(state['best_score_train']))
        else:
            assert env.cvfolds is not None

    def callback(env):
        """internal function"""
        score_train = env.evaluation_result_list[0][1]
        score = env.evaluation_result_list[1][1]
        if not state:
            init(env)
        best_score = state['best_score']
        best_iteration = state['best_iteration']
        maximize_score = state['maximize_score']
        if (maximize_score and score > best_score and ((not symetric_overfit and score_train - score <= max_overfit) or
                                                       (symetric_overfit and abs(score_train - score) <= max_overfit))) or \
                (not maximize_score and score < best_score and ((not symetric_overfit and score - score_train <= max_overfit) or
                                                                (symetric_overfit and abs(score - score_train) <= max_overfit))):
            msg = '[%d]\t%s' % (
                env.iteration,
                '\t'.join([_fmt_metric(x) for x in env.evaluation_result_list]))
            state['best_msg'] = msg
            state['best_score'] = score
            state['best_score_train'] = score_train
            state['best_iteration'] = env.iteration
            # save the property to attributes, so they will occur in checkpoint.
            if env.model is not None:
                env.model.set_attr(best_score=str(state['best_score']),
                                   best_score_train=str(state['best_score_train']),
                                   best_iteration=str(state['best_iteration']),
                                   best_msg=state['best_msg'])
        elif env.iteration - best_iteration >= stopping_rounds:
            best_msg = state['best_msg']
            if verbose and env.rank == 0:
                msg = "Stopping. Best iteration:\n{}\n\n"
                rabit.tracker_print(msg.format(best_msg))
            raise EarlyStopException(best_iteration)
    return callback

def train_xgb(params, dtrain, num_boost_round=10, evals=(), obj=None, feval=None,
          maximize=False, early_stopping_rounds=None, max_overfit = 0.02, symetric_overfit=False, evals_result=None,
          verbose_eval=True, xgb_model=None, callbacks=None, learning_rates=None):
    # pylint: disable=too-many-statements,too-many-branches, attribute-defined-outside-init
    """Train a booster with given parameters.
    Parameters
    ----------
    params : dict
        Booster params.
    dtrain : DMatrix
        Data to be trained.
    num_boost_round: int
        Number of boosting iterations.
    evals: list of pairs (DMatrix, string)
        List of items to be evaluated during training, this allows user to watch
        performance on the validation set.
    obj : function
        Customized objective function.
    feval : function
        Customized evaluation function.
    maximize : bool
        Whether to maximize feval.
    early_stopping_rounds: int
        Activates early stopping. Validation error needs to decrease at least
        every **early_stopping_rounds** round(s) to continue training.
        Requires at least one item in **evals**.
        If there's more than one, will use the last.
        Returns the model from the last iteration (not the best one).
        If early stopping occurs, the model will have three additional fields:
        ``bst.best_score``, ``bst.best_iteration`` and ``bst.best_ntree_limit``.
        (Use ``bst.best_ntree_limit`` to get the correct value if
        ``num_parallel_tree`` and/or ``num_class`` appears in the parameters)
    evals_result: dict
        This dictionary stores the evaluation results of all the items in watchlist.
        Example: with a watchlist containing
        ``[(dtest,'eval'), (dtrain,'train')]`` and
        a parameter containing ``('eval_metric': 'logloss')``,
        the **evals_result** returns
        .. code-block:: python
            {'train': {'logloss': ['0.48253', '0.35953']},
             'eval': {'logloss': ['0.480385', '0.357756']}}
    verbose_eval : bool or int
        Requires at least one item in **evals**.
        If **verbose_eval** is True then the evaluation metric on the validation set is
        printed at each boosting stage.
        If **verbose_eval** is an integer then the evaluation metric on the validation set
        is printed at every given **verbose_eval** boosting stage. The last boosting stage
        / the boosting stage found by using **early_stopping_rounds** is also printed.
        Example: with ``verbose_eval=4`` and at least one item in **evals**, an evaluation metric
        is printed every 4 boosting stages, instead of every boosting stage.
    learning_rates: list or function (deprecated - use callback API instead)
        List of learning rate for each boosting round
        or a customized function that calculates eta in terms of
        current number of round and the total number of boosting round (e.g. yields
        learning rate decay)
    xgb_model : file name of stored xgb model or 'Booster' instance
        Xgb model to be loaded before training (allows training continuation).
    callbacks : list of callback functions
        List of callback functions that are applied at end of each iteration.
        It is possible to use predefined callbacks by using
        :ref:`Callback API <callback_api>`.
        Example:
        .. code-block:: python
            [xgb.callback.reset_learning_rate(custom_rates)]
    Returns
    -------
    Booster : a trained booster model
    """
    callbacks = [] if callbacks is None else callbacks

    # Most of legacy advanced options becomes callbacks
    if isinstance(verbose_eval, bool) and verbose_eval:
        callbacks.append(callback.print_evaluation())
    else:
        if isinstance(verbose_eval, int):
            callbacks.append(callback.print_evaluation(verbose_eval))

    if early_stopping_rounds is not None:
        callbacks.append(early_stop_s(early_stopping_rounds,
                                               max_overfit,
                                            symetric_overfit=symetric_overfit,
                                             maximize=maximize,
                                             verbose=bool(verbose_eval)))
    if evals_result is not None:
        callbacks.append(callback.record_evaluation(evals_result))

    if learning_rates is not None:
        warnings.warn("learning_rates parameter is deprecated - use callback API instead",
                      DeprecationWarning)
        callbacks.append(callback.reset_learning_rate(learning_rates))

    return _train_internal(params, dtrain,
                           num_boost_round=num_boost_round,
                           evals=evals,
                           obj=obj, feval=feval,
                           xgb_model=xgb_model, callbacks=callbacks)

################################################## Modified hyperopt ###################################################
logger = logging.getLogger(__name__)

try:
    import cloudpickle as pickler
except Exception as e:
    logger.info('Failed to load cloudpickle, try installing cloudpickle via "pip install cloudpickle" for enhanced pickling support.')
    import six.moves.cPickle as pickler

class FMinIter(object):
    """Object for conducting search experiments.
    """
    catch_eval_exceptions = False
    pickle_protocol = -1

    def __init__(self, algo, domain, trials, rstate,
                 early_stop_round_mode_fun, early_stop_round,
                 asynchronous=None,
                 max_queue_len=1,
                 poll_interval_secs=1.0,
                 max_evals=10,
                 verbose=0,
                 show_progressbar=True
                 ):
        self.algo = algo
        self.domain = domain
        self.early_stop_round_mode_fun = early_stop_round_mode_fun
        self.early_stop_round = early_stop_round
        self.trials = trials
        self.show_progressbar = show_progressbar
        if asynchronous is None:
            self.asynchronous = trials.asynchronous
        else:
            self.asynchronous = asynchronous
        self.poll_interval_secs = poll_interval_secs
        self.max_queue_len = max_queue_len
        self.max_evals = max_evals
        self.rstate = rstate
        self.best_loss = np.inf

        if early_stop_round is not None:
            self.use_early_stop_round = True
            self.early_stop_iter = 1
        else:
            self.use_early_stop_round = False

        if self.asynchronous:
            if 'FMinIter_Domain' in trials.attachments:
                logger.warn('over-writing old domain trials attachment')
            msg = pickler.dumps(domain)
            # -- sanity check for unpickling
            pickler.loads(msg)
            trials.attachments['FMinIter_Domain'] = msg

    def serial_evaluate(self, N=-1):
        for trial in self.trials._dynamic_trials:
            if trial['state'] == base.JOB_STATE_NEW:
                trial['state'] = base.JOB_STATE_RUNNING
                now = coarse_utcnow()
                trial['book_time'] = now
                trial['refresh_time'] = now
                spec = base.spec_from_misc(trial['misc'])
                ctrl = base.Ctrl(self.trials, current_trial=trial)
                try:
                    result = self.domain.evaluate(spec, ctrl)
                except Exception as e:
                    logger.info('job exception: %s' % str(e))
                    trial['state'] = base.JOB_STATE_ERROR
                    trial['misc']['error'] = (str(type(e)), str(e))
                    trial['refresh_time'] = coarse_utcnow()
                    if not self.catch_eval_exceptions:
                        # -- JOB_STATE_ERROR means this trial
                        #    will be removed from self.trials.trials
                        #    by this refresh call.
                        self.trials.refresh()
                        raise
                else:
                    trial['state'] = base.JOB_STATE_DONE
                    trial['result'] = result
                    trial['refresh_time'] = coarse_utcnow()
                N -= 1
                if N == 0:
                    break
        self.trials.refresh()

    @property
    def is_cancelled(self):
        """
        Indicates whether this fmin run has been cancelled.  SparkTrials supports cancellation.
        """
        if hasattr(self.trials, "_fmin_cancelled"):
            if self.trials._fmin_cancelled:
                return True
        return False

    def block_until_done(self):
        already_printed = False
        if self.asynchronous:
            unfinished_states = [base.JOB_STATE_NEW, base.JOB_STATE_RUNNING]

            def get_queue_len():
                return self.trials.count_by_state_unsynced(unfinished_states)

            qlen = get_queue_len()
            while qlen > 0:
                if not already_printed:
                    logger.info('Waiting for %d jobs to finish ...' % qlen)
                    already_printed = True
                time.sleep(self.poll_interval_secs)
                qlen = get_queue_len()
            self.trials.refresh()
        else:
            self.serial_evaluate()

    def run(self, N, block_until_done=True):
        """
        block_until_done  means that the process blocks until ALL jobs in
        trials are not in running or new state

        """
        trials = self.trials
        algo = self.algo
        n_queued = 0

        def get_queue_len():
            return self.trials.count_by_state_unsynced(base.JOB_STATE_NEW)

        stopped = False
        qlen = get_queue_len()
        with std_out_err_redirect_tqdm() as orig_stdout:
            with tqdm(total=N+qlen, file=orig_stdout, postfix='best loss: ?',
                      disable=not self.show_progressbar, dynamic_ncols=True,
                      ) as pbar:
                while n_queued < N:
                    qlen = get_queue_len()
                    while qlen < self.max_queue_len and n_queued < N and not self.is_cancelled:
                        n_to_enqueue = min(self.max_queue_len - qlen, N - n_queued)
                        new_ids = trials.new_trial_ids(n_to_enqueue)
                        self.trials.refresh()
                        if 0:
                            for d in self.trials.trials:
                                print('trial %i %s %s' % (d['tid'], d['state'],
                                                          d['result'].get('status')))
                        new_trials = algo(new_ids, self.domain, trials,
                                          self.rstate.randint(2 ** 31 - 1))
                        assert len(new_ids) >= len(new_trials)
                        if len(new_trials):
                            self.trials.insert_trial_docs(new_trials)
                            self.trials.refresh()
                            n_queued += len(new_trials)
                            qlen = get_queue_len()
                        else:
                            stopped = True
                            break

                    if self.asynchronous:
                        # -- wait for workers to fill in the trials
                        time.sleep(self.poll_interval_secs)
                    else:
                        # -- loop over trials and do the jobs directly
                        self.serial_evaluate()

                    try:
                        # update best iteration
                        if self.trials.trials[len(self.trials.trials)-1]['result']['loss'] < self.best_loss:
                            self.best_loss = self.trials.trials[len(self.trials.trials) - 1]['result']['loss']
                            self.best_iter = n_queued

                            if self.use_early_stop_round:
                                self.early_stop_iter += 1
                                self.cur_early_stop_round = self.early_stop_round_mode_fun(self.early_stop_iter)*self.early_stop_round

                        # check early stop condition
                        if self.use_early_stop_round:
                            if n_queued - self.best_iter > self.cur_early_stop_round:
                                self.trials._fmin_cancelled = n_queued
                                self.trials._cur_early_stop_round = self.cur_early_stop_round
                                print('Early stoped at iteration %i' % n_queued+1)

                        pbar.postfix = 'best loss: %s, iteration number %i' % (str(self.best_loss), n_queued+1)
                    except:
                        pass
                    pbar.update(qlen)

                    if self.is_cancelled:
                        break

                    if stopped:
                        break

        if block_until_done:
            self.block_until_done()
            self.trials.refresh()
            logger.info('Queue empty, exiting run.')
        else:
            qlen = get_queue_len()
            if qlen:
                msg = 'Exiting run, not waiting for %d jobs.' % qlen
                logger.info(msg)

    def __iter__(self):
        return self

    def __next__(self):
        self.run(1, block_until_done=self.asynchronous)
        if len(self.trials) >= self.max_evals:
            raise StopIteration()
        return self.trials

    def exhaust(self):
        n_done = len(self.trials)
        self.run(self.max_evals - n_done, block_until_done=self.asynchronous)
        self.trials.refresh()
        return self

def fmin(fn, space, algo, max_evals, early_stop_round_mode_fun=None, early_stop_round=None, trials=None, rstate=None,
         allow_trials_fmin=False, pass_expr_memo_ctrl=None,
         catch_eval_exceptions=False,
         verbose=0,
         return_argmin=True,
         points_to_evaluate=None,
         max_queue_len=1,
         show_progressbar=True,
         ):
    """Minimize a function over a hyperparameter space.

    More realistically: *explore* a function over a hyperparameter space
    according to a given algorithm, allowing up to a certain number of
    function evaluations.  As points are explored, they are accumulated in
    `trials`


    Parameters
    ----------

    fn : callable (trial point -> loss)
        This function will be called with a value generated from `space`
        as the first and possibly only argument.  It can return either
        a scalar-valued loss, or a dictionary.  A returned dictionary must
        contain a 'status' key with a value from `STATUS_STRINGS`, must
        contain a 'loss' key if the status is `STATUS_OK`. Particular
        optimization algorithms may look for other keys as well.  An
        optional sub-dictionary associated with an 'attachments' key will
        be removed by fmin its contents will be available via
        `trials.trial_attachments`. The rest (usually all) of the returned
        dictionary will be stored and available later as some 'result'
        sub-dictionary within `trials.trials`.

    space : hyperopt.pyll.Apply node
        The set of possible arguments to `fn` is the set of objects
        that could be created with non-zero probability by drawing randomly
        from this stochastic program involving involving hp_<xxx> nodes
        (see `hyperopt.hp` and `hyperopt.pyll_utils`).

    algo : search algorithm
        This object, such as `hyperopt.rand.suggest` and
        `hyperopt.tpe.suggest` provides logic for sequential search of the
        hyperparameter space.

    max_evals : int
        Allow up to this many function evaluations before returning.

    trials : None or base.Trials (or subclass)
        Storage for completed, ongoing, and scheduled evaluation points.  If
        None, then a temporary `base.Trials` instance will be created.  If
        a trials object, then that trials object will be affected by
        side-effect of this call.

    rstate : numpy.RandomState, default numpy.random or `$HYPEROPT_FMIN_SEED`
        Each call to `algo` requires a seed value, which should be different
        on each call. This object is used to draw these seeds via `randint`.
        The default rstate is
        `numpy.random.RandomState(int(env['HYPEROPT_FMIN_SEED']))`
        if the `HYPEROPT_FMIN_SEED` environment variable is set to a non-empty
        string, otherwise np.random is used in whatever state it is in.

    verbose : int
        Print out some information to stdout during search.

    allow_trials_fmin : bool, default True
        If the `trials` argument

    pass_expr_memo_ctrl : bool, default False
        If set to True, `fn` will be called in a different more low-level
        way: it will receive raw hyperparameters, a partially-populated
        `memo`, and a Ctrl object for communication with this Trials
        object.

    return_argmin : bool, default True
        If set to False, this function returns nothing, which can be useful
        for example if it is expected that `len(trials)` may be zero after
        fmin, and therefore `trials.argmin` would be undefined.

    points_to_evaluate : list, default None
        Only works if trials=None. If points_to_evaluate equals None then the
        trials are evaluated normally. If list of dicts is passed then
        given points are evaluated before optimisation starts, so the overall
        number of optimisation steps is len(points_to_evaluate) + max_evals.
        Elements of this list must be in a form of a dictionary with variable
        names as keys and variable values as dict values. Example
        points_to_evaluate value is [{'x': 0.0, 'y': 0.0}, {'x': 1.0, 'y': 2.0}]

    max_queue_len : integer, default 1
        Sets the queue length generated in the dictionary or trials. Increasing this
        value helps to slightly speed up parallel simulatulations which sometimes lag
        on suggesting a new trial.

    show_progressbar : bool, default True
        Show a progressbar.

    Returns
    -------

    argmin : dictionary
        If return_argmin is True returns `trials.argmin` which is a dictionary.  Otherwise
        this function  returns the result of `hyperopt.space_eval(space, trails.argmin)` if there
        were succesfull trails. This object shares the same structure as the space passed.
        If there were no succesfull trails, it returns None.
    """
    if rstate is None:
        env_rseed = os.environ.get('HYPEROPT_FMIN_SEED', '')
        if env_rseed:
            rstate = np.random.RandomState(int(env_rseed))
        else:
            rstate = np.random.RandomState()

    if allow_trials_fmin and hasattr(trials, 'fmin'):
        return trials.fmin(
            fn, space,
            algo=algo,
            max_evals=max_evals,
            max_queue_len=max_queue_len,
            rstate=rstate,
            pass_expr_memo_ctrl=pass_expr_memo_ctrl,
            verbose=verbose,
            catch_eval_exceptions=catch_eval_exceptions,
            return_argmin=return_argmin,
            show_progressbar=show_progressbar,
        )

    if trials is None:
        if points_to_evaluate is None:
            trials = base.Trials()
        else:
            assert type(points_to_evaluate) == list
            trials = generate_trials_to_calculate(points_to_evaluate)

    domain = base.Domain(fn, space,
                         pass_expr_memo_ctrl=pass_expr_memo_ctrl)

    if early_stop_round is not None:
        # max_evals = 1000
        if early_stop_round_mode_fun is None:
            early_stop_round_mode_fun = lambda x: 1

    rval = FMinIter(algo, domain, trials, max_evals=max_evals,
                    early_stop_round_mode_fun = early_stop_round_mode_fun,
                    early_stop_round = early_stop_round,
                    rstate=rstate,
                    verbose=verbose,
                    max_queue_len=max_queue_len,
                    show_progressbar=show_progressbar)
    rval.catch_eval_exceptions = catch_eval_exceptions
    rval.exhaust()
    if return_argmin:
        if len(trials.trials) == 0:
            raise Exception("There are no evaluation tasks, cannot return argmin of task losses.")
        return trials.argmin
    elif len(trials) > 0:
        # Only if there are some succesfull trail runs, return the best point in the evaluation space
        return space_eval(space, trials.argmin)
    else:
        return None

################################################ Working with DataBase #################################################

def Table_to_DB(data, connection_string, table_name, to_string=True, grant_acces_to = 'public'):
    try:
        con = connect(connection_string)
        cur = con.cursor(cursor_factory=DictCursor)

        # Convert everything to string (Musthave due to bug in psycopg2 in saving floating numbers)
        data_ = data.copy(deep=True)
        if to_string:
            for i in data_.columns:
                data_[i] = data_[i].astype(str)

        sql = """
                   drop table if exists %(table_name)s; 

                   select *
                   into %(table_name)s
                   from (values %(values)s) as s(%(columns)s) 
               """ % {'table_name': table_name, 'values': '%s', 'columns': ', '.join(data_.columns)}

        execute_values(cur, sql, data_.values.tolist(), page_size=data_.shape[0])

        grant_acces_to = [grant_acces_to] if type(grant_acces_to) != list else grant_acces_to
        for u in grant_acces_to:
            sql = "grant select on %(table_name)s to %(user)s;" % {'table_name': table_name, 'user': u}
            cur.execute(sql)

        con.commit()
        con.close()

        return True
    except:
        return False


def Load_Table_from_DB(connection_string, table_name):
    try:
        con = connect(connection_string)
        cur = con.cursor(cursor_factory=DictCursor)

        cur.execute("""
                           select *
                           from
                               %s   
                       """ % table_name)
        data = pd.DataFrame(cur.fetchall(), columns=[cn[0] for cn in cur.description])

        con.close()

        return data
    except:
        return False

############################################### Additional PSI functions ###############################################

def PSI_for_monitoring(benchmark, data):
    """
        Calculate PSI with one predifined benchmark

        benchmark - DataFrame with obligatory columns [Max_bound, ]
        data - Series or List with current values of feature
    """

    data_ = pd.DataFrame({'Value': data, 'Cat': ''})
    benchmark_ = benchmark.reset_index()
    if benchmark_.loc[0, 'Max_bound'] != benchmark_.loc[0, 'Max_bound']: # Check that value is nan
        data_['Cat'] = data_['Value'].astype(str)
        benchmark_['Category'] = benchmark_['Category'].astype(str)
    else:
        cuts = [benchmark_.loc[0, 'Min_bound']] + benchmark_.loc[~benchmark_['Max_bound'].isna(), 'Max_bound'].tolist()
        data_['Cat'] = pd.cut(data_['Value'], bins=cuts, labels=pd.IntervalIndex.from_breaks(cuts).astype(str).to_list())
        data_['Cat'] = data_['Cat'].astype(str)


    data_.loc[data_['Value'].isna(), 'Cat'] = 'NA'
    benchmark_['Category'].fillna('NA', inplace=True)
    benchmark_['Category'].replace({'nan': 'NA'}, inplace=True)
    tmp = data_['Cat'].value_counts() / data_.shape[0]
    tmp = pd.DataFrame({'Category': tmp.index.astype(str), 'Cur_Dist': tmp.values})
    tmp = pd.merge(benchmark_[['Category', 'Pop_Dist']], tmp, how='left', on='Category')

    # Replace nan with 0 according to logic of nan in this case
    tmp.fillna(0, inplace=True)

    rez = core_PSI(tmp['Cur_Dist'], tmp['Pop_Dist'], mode='Limited')

    return rez

from numpy import log
def core_PSI(value1, value2, mode=['True', 'Limited'][0]):
    if mode == 'True':
        rez = (value1 - value2) * (log(value1 / value2))
        rez = sum(rez)
    elif mode == 'Limited':
        rez = 0
        for i in range(len(value1)):
            rez += (value1[i] - value2[i]) * min(max(log(value1[i] / value2[i]), -1), 1)
    else:
        rez = None

    # return result
    return rez

def Population_Distribution(df, var, nbins = 10):
    # check that var isn't all nulls column
    if df[var].isna().all():
        return None

    # cut numeric features
    if is_numeric(df[var]) and len(df[var].unique()) > nbins:
        if df[var].dtype == 'O':
            df[var] = pd.to_numeric(df[var])

        # Calculate cuts
        cuts = df[var].quantile(q=np.linspace(0, 1, nbins+1)).unique()
        cuts[0] = -np.inf
        cuts[-1] = np.inf

        # Calculate population distribution by cuts
        rez = pd.DataFrame({'Min_bound': cuts[:-1], 'Max_bound': cuts[1:], 'Category': pd.IntervalIndex.from_breaks(cuts).astype(str).to_list(),
                            'Pop_Dist': pd.cut(df[var], bins=pd.IntervalIndex.from_breaks(cuts), duplicates='drop').value_counts(sort=False)})

        # add NA distribution
        if df[var].isna().sum() > 0:
            rez = rez.append({'Min_bound': None, 'Max_bound': None, 'Category': 'NA', 'Pop_Dist': df[var].isna().sum()}, ignore_index=True)

    else:
        if df[var].dtype.name != 'category':
            df[var] = df[var].astype('category')

        if 'NA' not in df[var].cat.categories:
            df[var].cat.add_categories('NA', inplace=True)
        df[var].fillna('NA', inplace=True)

        rez = pd.DataFrame({'Min_bound': None, 'Max_bound': None, 'Category': df[var].cat.categories,
                            'Pop_Dist': df[var].value_counts(sort=False)})

    rez['Pop_Dist'] /= rez['Pop_Dist'].sum()

    return rez

################################################### Trees processing ###################################################

def get_leafs(tree, node=0, rez=[]):
    if tree.tree_.children_left[node] == -1:
        rez = rez + [node]
    else:
        rez = get_leafs(tree, tree.tree_.children_left[node], rez)
        rez = get_leafs(tree, tree.tree_.children_right[node], rez)

    return rez

def gradient_color(start, stop, n = None, saturation = None):
    red = int(start[1:3], 16)
    red_diff = int(stop[1:3], 16) - int(start[1:3], 16)
    green = int(start[3:5], 16)
    green_diff = int(stop[3:5], 16) - int(start[3:5], 16)
    blue = int(start[5:7], 16)
    blue_diff = int(stop[5:7], 16) - int(start[5:7], 16)
    result = []
    if n is not None:
        for i in range(n):
            result += ['#' +
                       '{:02X}'.format(round(red + red_diff * i / (n-1))) + \
                       '{:02X}'.format(round(green + green_diff * i / (n-1))) + \
                       '{:02X}'.format(round(blue + blue_diff * i / (n-1)))
                       ]
    elif saturation is not None:
        for i in saturation:
            result += ['#' +
                       '{:02X}'.format(int(red + red_diff * i)) + \
                       '{:02X}'.format(int(green + green_diff * i)) + \
                       '{:02X}'.format(int(blue + blue_diff * i))
                       ]

    return result

def prune_decline_tree(tree, leaf=0):
    if tree.tree_.children_right[leaf] != -1:
        right = prune_decline_tree(tree, leaf=tree.tree_.children_right[leaf])
        left = prune_decline_tree(tree, leaf=tree.tree_.children_left[leaf])
        if left == right and left != -1:
            tree.decline[leaf] = left
            tree.tree_.children_right[leaf] = -1
            tree.tree_.children_left[leaf] = -1

    return tree.decline[leaf]

def define_bounds(tree, leaf=0,  bounds=None):
    # generate root bounds
    if bounds is None:
        bounds = np.zeros((len(tree.feature_names), 2))
        bounds[:, 1] = 1

    tree.bounds[leaf] = bounds

    if tree.tree_.feature[leaf] !=-2:
        bt = bounds.copy()
        bt[tree.tree_.feature[leaf], 1] = min(bt[tree.tree_.feature[leaf], 1], tree.tree_.threshold[leaf])
        define_bounds(tree, tree.tree_.children_left[leaf], bt)
        bt = bounds.copy()
        bt[tree.tree_.feature[leaf], 0] = max(bt[tree.tree_.feature[leaf], 0], tree.tree_.threshold[leaf])
        define_bounds(tree, tree.tree_.children_right[leaf], bt)

############################################## Save / load models objects ##############################################

def save_model(path, model):
    try:
        # in case when script is running from source file
        model.code = inspect.getsource(model)
    except:
        # in case when script is running from console
        res = 'class %s:' % re.findall(r'.*\.(.*)\'', str(model.__class__))[0]
        for f in dir(model.__class__):
            if(type(getattr(model, f)) == types.MethodType):
                res += '\n' + inspect.getsource(getattr(model, f)) + '\n'

        model.code = res

    pickle.dump(model, open(path, "wb"))

def load_model(path):
    res = pickle.load(open(path, "rb"))
    exec(res.code)

    return res

################################################## Profit Tree Class ###################################################
class Profit_Tree:
    def fit(self, data, feat_cols, opt_col, min_cnt_in_leaf=100):
        self.cnt = data.shape[0]
        self.opt = data[opt_col].sum()
        self.div_feat = None
        val = 0

        for feat in feat_cols:
            if data[opt_col].sum() > 0:
                tmp = data[[feat, opt_col]].sort_values(feat)[opt_col].expanding().sum()[min_cnt_in_leaf:-min_cnt_in_leaf]
                if tmp.min() < val:
                    self.div_feat = feat
                    self.div_val = data.loc[max(pd.np.where(tmp == tmp.min(), tmp.index, -1)), feat]
                    val = tmp.min()
            else:
                tmp = data[[feat, opt_col]].sort_values(feat, ascending=False)[opt_col].expanding().sum()[min_cnt_in_leaf:-min_cnt_in_leaf]
                if tmp.max() > val:
                    self.div_feat = feat
                    self.div_val = data.loc[max(pd.np.where(tmp == tmp.max(), tmp.index, -1)), feat]
                    val = tmp.max()

        if self.div_feat is not None and data.shape[0]>200:
            self.left = Profit_Tree()
            self.left.fit(data.query('%s <= @self.div_val' % self.div_feat), feat_cols, opt_col, min_cnt_in_leaf)

            self.right = Profit_Tree()
            self.right.fit(data.query('%s >  @self.div_val' % self.div_feat), feat_cols, opt_col, min_cnt_in_leaf)

    def fit_avg_value(self, data, feat_cols, opt_col, avg_val, avg_val_bound, min_cnt_in_leaf=100):
        self.cnt = data.shape[0]
        self.opt = data[opt_col].mean()
        self.div_feat = None
        val = 0

        if self.opt + avg_val_bound < avg_val:
            for feat in feat_cols:
                tmp = abs(data[[feat, opt_col]].sort_values(feat, ascending=False)[opt_col].expanding().mean()[min_cnt_in_leaf:-min_cnt_in_leaf] - avg_val)
                if tmp.min() < avg_val_bound:
                    tmp = data.loc[max(np.where(tmp == tmp.min(), tmp.index, -1)), feat]

                    if sum(data[feat]>=tmp) > val:
                        self.div_feat = feat
                        self.div_val = tmp
                        val = sum(data[feat]>=tmp)

        if self.div_feat is not None and data.shape[0]>200:
            self.left = Profit_Tree()
            self.left.fit_avg_value(data.query('%s <= @self.div_val' % self.div_feat), feat_cols, opt_col, avg_val, avg_val_bound, min_cnt_in_leaf)

            self.right = Profit_Tree()
            self.right.fit_avg_value(data.query('%s >  @self.div_val' % self.div_feat), feat_cols, opt_col, avg_val, avg_val_bound, min_cnt_in_leaf)

    def print(self, condition=''):
        res = ['{} | cnt {:,.0f} | opt {:,.0f}'.format(condition, self.cnt, self.opt)]

        if self.div_feat is not None:
            tmp = self.left.print('{} <= {:,.3f}'.format(self.div_feat, self.div_val))
            res += ['|-->' + tmp[0]]
            if len(tmp)>1:
                res += ['|   ' + i for i in tmp[1:]]

            tmp = self.right.print('{} > {:,.3f}'.format(self.div_feat, self.div_val))
            res += ['|-->' + tmp[0]]
            if len(tmp) > 1:
                res += ['|   ' + i for i in tmp[1:]]

        return res

    def __repr__(self):
        res = self.print()
        if len(res) > 1:
            res[0] = 'ROOT' + res[0]
            res = '\n'.join(res)
        else:
            res = 'LEAF' + res[0]

        return res

    def get_val(self, data):

        if self.div_feat is not None:
            res = pd.Series(0, data.index)
            musk = data[self.div_feat] <= self.div_val
            res[musk] = self.left.get_val(data.loc[musk, :])
            res[~musk] = self.right.get_val(data.loc[~musk, :])

        else:
            res = pd.Series(self.opt, data.index)

        return res

    def set_leafs(self, leaf=0):
        self.leaf = leaf
        if self.div_feat is not None:
            leaf = self.left.set_leafs(leaf + 1)
            leaf = self.right.set_leafs(leaf + 1)

        return leaf

    def get_leaf(self, data):
        if not hasattr(self, 'leaf'):
            self.set_leafs()

        if self.div_feat is not None:
            res = pd.Series(0, data.index)
            musk = data[self.div_feat] <= self.div_val
            res[musk] = self.left.get_leaf(data.loc[musk, :])
            res[~musk] = self.right.get_leaf(data.loc[~musk, :])

        else:
            res = pd.Series(self.leaf, data.index)

        return res

    def set_bounds(self, bounds):
        self.bounds = bounds

        if self.div_feat is not None:
            tmp = bounds.copy()
            tmp.loc[self.div_feat, 'max'] = self.div_val
            self.left.set_bounds(tmp)
            tmp = bounds.copy()
            tmp.loc[self.div_feat, 'min'] = self.div_val
            self.right.set_bounds(tmp)

    def get_bounds(self):
        if self.div_feat is None:
            res = pd.np.array(self.bounds)
            res.resize((1, res.shape[0], res.shape[1]))
        else:
            res = pd.np.empty((0, self.bounds.shape[0], self.bounds.shape[1]))
            res = pd.np.append(res, self.left.get_bounds(), 0)
            res = pd.np.append(res, self.right.get_bounds(), 0)

        return res

