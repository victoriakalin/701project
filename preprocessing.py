import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn import preprocessing



T_F_COLS = ['school_charter', 'school_magnet', 'school_year_round', 'school_nlns',
       'school_kipp', 'school_charter_ready_promise','teacher_teach_for_america', 'teacher_ny_teaching_fellow','eligible_double_your_impact_match', 'eligible_almost_home_match', "high_poverty"]
CAT_COLS = ['school_state', 'school_metro', 'primary_focus_subject', 'primary_focus_area', 'resource_type', 'grade_level'] #, 'poverty_level']

def make_data(projects_fp = "kdd-cup-2014-predicting-excitement-at-donors-choose/projects.csv",
              outcomes_fp = "kdd-cup-2014-predicting-excitement-at-donors-choose/outcomes.csv",
              essays_fp = "kdd-cup-2014-predicting-excitement-at-donors-choose/essays.csv"):
    projects = pd.read_csv(projects_fp)
    essays = pd.read_csv(essays_fp)
    essays["has_essay"] = 1
    # essays = essays.drop( columns = ['teacher_acctid', 'title', 'short_description',
    #    'need_statement', 'essay'])
    # print("ESSYA COLS: ", essays.columns)
    # print("# COLS:", len(projects.columns))
    outcomes = pd.read_csv(outcomes_fp)
    # print(outcomes.columns)
    # print(projects.columns)
    # print("PROJ ROWS:", projects.shape[0])
    projects = pd.merge(projects, essays, on = "projectid", how = "left")
    # print("PROJ ROWS:", projects.shape[0])
    # print("Project  COLS: ", projects.columns)
    total = pd.merge(projects, outcomes, on = "projectid", how = "outer")
    # total = total[['projectid', 'fully_funded','school_latitude', 'school_longitude',
    #              'school_charter', 'school_magnet', 'school_year_round', 'school_nlns',
    #              'school_kipp', 'school_charter_ready_promise', 'teacher_prefix',
    #              'teacher_teach_for_america', 'teacher_ny_teaching_fellow',
    #              'primary_focus_subject', 'primary_focus_area', 'resource_type',
    #              'poverty_level', 'grade_level', 'fulfillment_labor_materials',
    #              'total_price_excluding_optional_support',
    #              'total_price_including_optional_support', 'students_reached',
    #              'eligible_double_your_impact_match', 'eligible_almost_home_match','school_state', 'school_metro', 'has_essay']]
    # print(total['teacher_prefix'].unique())
    # Make Teacher Gender
    print(total.columns)
    total = total.replace({"Mrs.": 1, "Mr.": 0, "Ms.": 1, "Mr. & Mrs.":float("nan"), "Dr.":float("nan")})
    total = total.rename(columns = {"teacher_prefix":"teacher_gender"})
    # Make Income Level
    total = total.replace({"highest poverty": 1, "high poverty": 1, "moderate poverty": 0, "low poverty":0})
    total = total.rename(columns = {"poverty_level":"high_poverty"})
    # print(total['teacher_gender'])
    print(total['fully_funded'].notna())
    total = total[['fully_funded', 'title', "projectid",'school_latitude', 'school_longitude',
                 'school_charter', 'school_magnet', 'school_year_round', 'school_nlns',
                 'school_kipp', 'school_charter_ready_promise', 'teacher_gender',
                 'teacher_teach_for_america', 'teacher_ny_teaching_fellow',
                 'primary_focus_subject', 'primary_focus_area', 'resource_type',
                 'high_poverty', 'grade_level', 'fulfillment_labor_materials',
                 'total_price_excluding_optional_support',
                 'total_price_including_optional_support', 'students_reached',
                 'eligible_double_your_impact_match', 'eligible_almost_home_match','school_state', 'school_metro', 'has_essay']]
    total = pd.get_dummies(total, columns=CAT_COLS)

    train_X = total[total['fully_funded'].notna()]
    train_X = train_X.replace({"t":1, "f":0})
    test = total[total['fully_funded'].isna()]
    test = test.replace({"t":1, "f":0})

    test.dropna()
    train_X = train_X.dropna()
    train_y = train_X[['fully_funded']]
    train_X = train_X.drop(["fully_funded", "projectid", "title"], axis = 1)


    print("DOne")
    print(train_X.head())
    print(train_X.columns)
    print(test.columns)
    # train_X = preprocessing.scale(train_X)
    return test, train_X, train_y


test, train_X, train_y = make_data()
print(np.shape(test))
print(np.shape(train_X))
print(np.shape(train_y))

print("returned")
with open('test_set.pickle', 'wb') as f:
    pickle.dump(test, f)
print("wrote test")
with open('train_X.pickle', 'wb') as f:
    pickle.dump(train_X, f, protocol=4)
print("wrote train X")
with open('train_y_set.pickle',  'wb') as f:
    pickle.dump(train_y, f)