#!python3
"""
Main file for analyzing the persistence of students in college.

Inputs (all as csv files in inputs folder):
    Senior Survey Data (has text based responses w/ some blanks)
    Senior Survey Key (decodes the responses above)
    Alumni Persistence
    Trials specification (which variables for regression trials
"""

import pandas as pd
import numpy as np
from modules.predictions import Prediction

def map_numbers(raw_answer, mapping_dict):
    '''applied function to map a raw survey answer to a number'''
    if pd.isnull(raw_answer):
        return np.NaN
    if raw_answer in mapping_dict:
        return mapping_dict[raw_answer]
    else:
        return -1

def process_survey_file(survey_data_file, survey_key_file):
    """performs analysis on survey data"""
    sdf = pd.read_csv(survey_data_file,encoding='cp1252', index_col=0)
    skf = pd.read_csv(survey_key_file,encoding='cp1252')

    # Develop question groupings
    questions = list(skf.Key)
    question_groups = list(set([q[:-1] if q[-1].isdigit() else q
                                for q in questions]))
    question_hierarchy = {qg:[q for q in questions if q.startswith(qg)]
                         for qg in question_groups}

    # Map numeric values to each question
    map_d = {}
    for i in range(len(skf)):
        item = skf.loc[i].tolist()
        item_d = {}
        for j in range(1, len(item)):
            ref = int(item[j]) if item[j].isdigit() else item[j]
            item_d[ref] = j # this works because column headers are 1-5
        map_d[item[0]] = item_d

    answers_df = sdf.copy()
    for i in range(2,len(answers_df.iloc[0])):
        answers_df.iloc[:,i] = answers_df.iloc[:,i].apply(
                map_numbers, args=[map_d[answers_df.columns[i]]])

    answers_df.to_csv('scored_answers.csv')
    answers_df.describe().to_csv('scored_answers_stats.csv')

    # Collapse scores into groups for each student
    group_scores = sdf.iloc[:,:1].copy()
    for group, qs in question_hierarchy.items():
        these_qs = answers_df[qs]
        this_group = these_qs.mean(axis=1)
        this_group.name = group
        group_scores = pd.concat([group_scores, this_group], axis=1)

    group_scores.to_csv('grouped_answers.csv')
    group_scores.describe().to_csv('grouped_answers_stats.csv')

    return group_scores

def main(survey_data_file, survey_key_file, persistence_file, trial_file):
    survey_df = process_survey_file(survey_data_file, survey_key_file)
    main_df = pd.read_csv(persistence_file, encoding='cp1252', index_col=0)
    main_df = pd.concat([main_df, survey_df], axis=1)
    main_df.to_csv('combined_input_data.csv')

    newP = Prediction(main_df,
            [2012, 2013, 2014, 2015],
            ['GPA', 'ACT', 'Initial PGR'],
            'Retention3', 'GPA, ACT, and GR for 2012-2015 Males',
            [('IsMale', 0)]
            )
    newP.to_csv('pred_test.csv')
    newP.describe()




if __name__ == '__main__':
    main(
            'inputs/Senior_Survey_Data.csv',
            'inputs/Senior_Survey_Key.csv',
            'inputs/Persistence_Data.csv',
            'inputs/Trials_Specs.csv'
            )
