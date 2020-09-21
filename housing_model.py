import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import LinearSVR



def read_csv(filepath):

    housing = pd.read_csv(filepath + 'housing.csv')

    return housing

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

def DecisionTree_predictions(X_train,y_train):
	#TODO: complete this
    tree_reg = DecisionTreeRegressor(max_depth = 1)
    #tree_reg2 = AdaBoostRegressor(tree_reg1, n_estimators= 1000).fit(X_train,Y_train)
    tree_reg.fit(X_train, y_train)
    return tree_reg, tree_reg.predict(X_train)
    

def AdaBoost_predictions(X_train,y_train):
	#TODO: complete this
    
    ada_clf = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=1), n_estimators=200)
    ada_clf.fit(X_train, y_train)
    return ada_clf.predict(X_train)


def main():
    SHOW_PRINT = True
    path = './data/'
    housing =  read_csv(path)
    if SHOW_PRINT:
        print(housing.head())
        housing.hist(bins=50, figsize=(20,15))
    train_set, test_set = split_train_test(housing, 0.2)
    if SHOW_PRINT:
        print(len(train_set))
    #print(plt.save_fig("attribute_histogram_plots"))
        print(housing["median_income"].hist())
    
    housing_with_id = housing.reset_index() 
    if SHOW_PRINT:
        print(plt.show())
    # adds an `index` column
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
    housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
    if SHOW_PRINT:
        print(test_set.head())
        print(housing["median_income"].hist())
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
    housing["income_cat"].value_counts()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    
    if SHOW_PRINT:    
        print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
    
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
    }).sort_index()
    compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
    compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    
    if SHOW_PRINT:
        print(compare_props)
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    housing = strat_train_set.copy()
    
    if SHOW_PRINT:
        plt.legend()
    
    attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
    if SHOW_PRINT:
        scatter_matrix(housing[attributes], figsize=(12, 8))
    
    
    if SHOW_PRINT:   
        housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    
    housing = strat_train_set.drop("median_house_value", axis=1)
    
    housing_labels = strat_train_set["median_house_value"].copy()
    
    
    # Data clean    
    median = housing["total_bedrooms"].median()  
    housing["total_bedrooms"].fillna(median, inplace=True)
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room
        def fit(self, X, y=None):
            return self  # nothing else to do
        def transform(self, X):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household,                         bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing.values)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    housing_num_tr = num_pipeline.fit_transform(housing_num)


    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)


    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data) 
    print("Predictions:", lin_reg.predict(some_data_prepared))
    housing_predictions = lin_reg.predict(housing_prepared)
    print("Labels:", list(some_labels))
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    # print result of linear regression
    print("Median Predicted House Value",lin_rmse)
    tree_reg,tree_pred = DecisionTree_predictions(housing_prepared,housing_labels)
    lin_mse = mean_squared_error(housing_labels, tree_pred)
    lin_rmse = np.sqrt(lin_mse)
    print("Median Predicted House Value using Decision Tree",lin_rmse)
    
    tree_pred = AdaBoost_predictions(housing_prepared,housing_labels)
    lin_mse = mean_squared_error(housing_labels, tree_pred)
    lin_rmse = np.sqrt(lin_mse)
    print("Median Predicted House Value using AdaBoost_predictions",lin_rmse)
    
    svm_reg = LinearSVR(epsilon=1.5)
    svm_reg.fit(housing_prepared, housing_labels)
    lin_mse = mean_squared_error(housing_labels, tree_pred)
    lin_rmse = np.sqrt(lin_mse)
    print("Median Predicted House Value using LinearSVR_predictions",lin_rmse) 
    
    
    
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    if SHOW_PRINT: 
        display_scores(tree_rmse_scores)
        print(tree_reg.predict(housing_prepared))

    
#    final_model = tree_reg
#
#    X_test = strat_test_set.drop("median_house_value", axis=1)
#    y_test = strat_test_set["median_house_value"].copy()
#
#    X_test_prepared = full_pipeline.transform(X_test)
#
#    final_predictions = final_model.predict(X_test_prepared)
#
#    final_mse = mean_squared_error(y_test, final_predictions)
#    final_rmse = np.sqrt(final_mse)
#    print(final_rmse)
    
if __name__ == "__main__":
    main()

	