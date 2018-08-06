from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV,train_test_split
import xgboost as xgb
import lightgbm as lgb

#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train.values, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

class grid():
    def __init__(self,model):
        self.model = model
    
    def grid_get(self,param_grid):
        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
        grid_search = GridSearchCV(self.model,param_grid,cv=kf, scoring="neg_mean_squared_error", n_jobs=-1,verbose=1)
        grid_search.fit(train.values, y_train.values)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])


grid(RandomForestRegressor()).grid_get({'n_estimators': [100,200],'max_features':[0.1,0.5],'max_depth':[10,50],'min_samples_split':[2,10],'min_samples_leaf':[2,10]})