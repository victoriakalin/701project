import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle


T_F_COLS = ['school_charter', 'school_magnet', 'school_year_round', 'school_nlns',
       'school_kipp', 'school_charter_ready_promise','teacher_teach_for_america', 'teacher_ny_teaching_fellow','eligible_double_your_impact_match', 'eligible_almost_home_match']
CAT_COLS = ['school_state', 'school_metro', 'school_county', 'primary_focus_subject', 'primary_focus_area', 'resource_type', 'poverty_level', 'grade_level']

def make_data(projects_fp = "kdd-cup-2014-predicting-excitement-at-donors-choose/projects.csv", outcomes_fp = "kdd-cup-2014-predicting-excitement-at-donors-choose/outcomes.csv"):
    projects = pd.read_csv(projects_fp)
    outcomes = pd.read_csv(outcomes_fp)
    # print(outcomes.columns)
    # print(projects.columns)
    total = pd.merge(projects, outcomes, on = "projectid", how = "outer")
    total = total[['fully_funded','school_latitude', 'school_longitude', 'school_state',
                 'school_metro', 'school_county', 'school_charter', 'school_magnet', 'school_year_round', 'school_nlns',
                 'school_kipp', 'school_charter_ready_promise', 'teacher_prefix',
                 'teacher_teach_for_america', 'teacher_ny_teaching_fellow',
                 'primary_focus_subject', 'primary_focus_area', 'resource_type',
                 'poverty_level', 'grade_level', 'fulfillment_labor_materials',
                 'total_price_excluding_optional_support',
                 'total_price_including_optional_support', 'students_reached',
                 'eligible_double_your_impact_match', 'eligible_almost_home_match']]
    print(total['teacher_prefix'].unique())
    # print(1/0)
    total = total.replace({"Mrs.": 1, "Mr.": 0, "Ms.": 1, "Mr. & Mrs.":float("nan"), "Dr.":float("nan")})
    total = total.rename(columns = {"teacher_prefix":"teacher_gender"})
    # print(total['teacher_gender'])
    total = pd.get_dummies(total, columns=CAT_COLS)
    print(total.columns)
    train = total[total['fully_funded'].notna()]
    train = train.replace({"t":1, "f":0})
    test = total[total['fully_funded'].isna()]
    test = test.replace({"t":1, "f":0})
    test = test.dropna()
    test.drop(['fully_funded'], axis = 1)
    train = train.dropna()
    train_y = train[['fully_funded']]
    train_X = train.drop(['fully_funded'], axis = 1)
    print("DOne")
    print(train_X.head())
    print(train_X.columns)
    print(test.columns)
    return test, train_X, train_y



test, train_X, train_y= make_data()
print("returned")
with open('test_set.csv.pickle', 'wb') as f:
    pickle.dump(test, f)
print("wrote test")
with open('train_X.pickle', 'wb') as f:
    pickle.dump(train_X, f)
print("wrote train X")
with open('train_y_set.pickle',  'wb') as f:
    pickle.dump(train_y, f)