import pandas as pd
import numpy as np

def make_data(projects_fp = "kdd-cup-2014-predicting-excitement-at-donors-choose/projects.csv", outcomes_fp = "kdd-cup-2014-predicting-excitement-at-donors-choose/outcomes.csv"):
    projects = pd.read_csv(projects_fp)
    outcomes = pd.read_csv(outcomes_fp)
    # print(outcomes.columns)
    # print(projects.columns)
    total = pd.merge(projects, outcomes, on = "projectid", how = "outer")
    train = total[total['fully_funded'].notna()]
    test = total[total['fully_funded'].isna()]
    test = test[['projectid', 'teacher_acctid', 'schoolid', 'school_ncesid',
       'school_latitude', 'school_longitude', 'school_city', 'school_state',
       'school_zip', 'school_metro', 'school_district', 'school_county',
       'school_charter', 'school_magnet', 'school_year_round', 'school_nlns',
       'school_kipp', 'school_charter_ready_promise', 'teacher_prefix',
       'teacher_teach_for_america', 'teacher_ny_teaching_fellow',
       'primary_focus_subject', 'primary_focus_area',
       'secondary_focus_subject', 'secondary_focus_area', 'resource_type',
       'poverty_level', 'grade_level', 'fulfillment_labor_materials',
       'total_price_excluding_optional_support',
       'total_price_including_optional_support', 'students_reached',
       'eligible_double_your_impact_match', 'eligible_almost_home_match',
       'date_posted']]
    return test, train



test, train = make_data()
test.to_csv("test_set.csv")
train.to_csv('train_set.csv')