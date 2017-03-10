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

def process_survey_file(survey_data_file, survey_key_file, save_details=False):
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

    if save_details:
        answers_df.to_csv('scored_answers.csv')
        answers_df.describe().to_csv('scored_answers_stats.csv')

    # Collapse scores into groups for each student
    group_scores = sdf.iloc[:,:1].copy()
    for group, qs in question_hierarchy.items():
        these_qs = answers_df[qs]
        this_group = these_qs.mean(axis=1)
        this_group.name = group
        group_scores = pd.concat([group_scores, this_group], axis=1)

    if save_details:
        group_scores.to_csv('grouped_answers.csv')
        group_scores.describe().to_csv('grouped_answers_stats.csv')

    return group_scores

def main(survey_data_file, survey_key_file, persistence_file, trial_file,
        gpa, outcome):
    survey_df = process_survey_file(survey_data_file, survey_key_file)
    main_df = pd.read_csv(persistence_file, encoding='cp1252', index_col=0)
    main_df = pd.concat([main_df, survey_df], axis=1)
    main_df.to_csv('combined_input_data.csv')

    special_colleges = (
            (145600, 'University of Illinois at Chicago'),
            (145637, 'University of Illinois at Urbana-Champaign'),
            (149772, 'Western Illinois University'),
            (144209, 'City Colleges of Chicago-Harold Washington College'),
            (145813, 'Illinois State University'),
            (147776, 'Northeastern Illinois University'),
            (144218, 'City Colleges of Chicago-Wilbur Wright College'),
            (170301, 'Hope College'),
            (149222, 'Southern Illinois University Carbondale'),
            (148654, 'University of Illinois at Springfield'),
            (147341, 'Monmouth College'),
            (144892, 'Eastern Illinois University'),
            (144740, 'DePaul University'),
            (148496, 'Dominican University'),
            (147703, 'Northern Illinois University'),
            )
    special_exclude = [str(x[0]) for x in special_colleges]
    results_dfs = []
    coefs_dfs = []
    # first do a few sample cases 2013-2014 testing on 2015
    for require, remove, text in [
            (None, None, "GPA/GR for '13-14 ('15 test)"),
            (None, [('Initial NCES', special_exclude)],
                            "GPA/GR for '13-14 ('15 test) no big c"),
            ]:
        newP = Prediction(main_df, [2013, 2014, 2015],
                [gpa, 'Initial PGR',],
            outcome, text, require=require, remove=remove,
            train=[2013, 2014])
        results_dfs.append(newP.desc_df)
        newP = Prediction(main_df, [2013, 2014, 2015],
                [gpa, 'Initial PGR', 'IsMale', 'IsBlack', 'IsLatino',
                    'IsSpEd'],
            outcome, text+' (plus demo)', require=require, remove=remove,
            train=[2013, 2014])
        results_dfs.append(newP.desc_df)
        coefs_dfs.append(newP.make_coef_df(text))

    # do sample cases for big colleges
    for nces, school_name in special_colleges:
        requireA = ('Initial NCES', str(nces))
        text = "GPA for '13-14 ('15 test) at "+school_name
        newP = Prediction(main_df, [2013, 2014, 2015], [gpa],
            outcome, text, require=[requireA], train=[2013, 2014])
        results_dfs.append(newP.desc_df)
        newP = Prediction(main_df, [2013, 2014, 2015],
                [gpa, 'IsMale', 'IsBlack', 'IsLatino'],
            outcome, text+' (plus demo)',
            require=[requireA], train=[2013, 2014])
        results_dfs.append(newP.desc_df)
        #coefs_dfs.append(newP.make_coef_df(nces))
        newP = Prediction(main_df, [2013, 2014, 2015],
                [gpa, 'IsMale', 'IsBlack', 'IsLatino'],
            outcome, text+' (plus demo+2015)',
            require=[requireA], train=None)
        results_dfs.append(newP.desc_df)
        coefs_dfs.append(newP.make_coef_df(nces))

    # Do cases for each Barron's category (no big colleges)
    barrons_cases = [
            ('IsMCPlus', 'Most Competitive+'),
            ('IsMC', 'Most Competitive'),
            ('IsHC', 'Highly Competitive'),
            ('IsVC', 'Very Competitive'),
            ('IsC', 'Competitive'),
            ('IsNC', 'Noncompetitive'),
            ('Is2yr', '2 year'),
            ]
    for field, label in barrons_cases:
        newP = Prediction(main_df, [2013, 2014, 2015],
                [gpa, 'Initial PGR', 'IsMale', 'IsBlack', 'IsLatino'],
                outcome, label+' (plus demo no big c)',
                require=[(field, 1)],
                remove=[('Initial NCES', special_exclude)],
                train=[2013, 2014])
        results_dfs.append(newP.desc_df)
        #coefs_dfs.append(newP.make_coef_df(label))
        newP = Prediction(main_df, [2013, 2014, 2015],
                [gpa, 'Initial PGR', 'IsMale', 'IsBlack', 'IsLatino'],
                outcome, label+' (plus demo no big c +2015)',
                require=[(field, 1)],
                remove=[('Initial NCES', special_exclude)],
                train=None)
        results_dfs.append(newP.desc_df)
        coefs_dfs.append(newP.make_coef_df(label))

    # look at HBCUs
    newP = Prediction(main_df, [2012, 2013, 2014, 2015],
            [gpa, 'Initial PGR', 'IsMale'],
            outcome,
            '2012-15 HBCU', require=[('IsInitialHBCU',1)], remove=None,
            train=None)
    results_dfs.append(newP.desc_df)
    coefs_dfs.append(newP.make_coef_df('2012-15 HBCU'))

    # look at 2015 with senior survey
    newP = Prediction(main_df, [2015],
            [gpa, 'Initial PGR'], outcome,
            '2015 base GPA/GR', require=None, remove=None,
            train=None)
    results_dfs.append(newP.desc_df)
    newP = Prediction(main_df, [2015],
            [gpa, 'Initial PGR', 'IsMale', 'IsBlack', 'IsLatino'],
            outcome,
            '2015 base GPA/GR w demo', require=None, remove=None,
            train=None)
    results_dfs.append(newP.desc_df)
    newP = Prediction(main_df, [2015],
            [gpa, 'Initial PGR', 'IsMale', 'IsBlack', 'IsLatino'],
            outcome,
            '2015 base GPA/GR w demo; no big c', require=None,
            remove=[('Initial NCES', special_exclude)],
            train=None)
    results_dfs.append(newP.desc_df)
    newP = Prediction(main_df, [2015],
            [gpa, 'Initial PGR', 'IsMale', 'IsBlack', 'IsLatino',
                'Academic_Identity'],
            outcome,
            '2015 base GPA/GR w demo; 1 survey', require=None,remove=None,
            train=None)
    results_dfs.append(newP.desc_df)
    newP = Prediction(main_df, [2015],
            [gpa, 'Initial PGR', 'IsMale', 'IsBlack', 'IsLatino',
                'Academic_Identity'],
            outcome,
            '2015 base GPA/GR w demo; 1 survey_test', require=None,remove=None,
            train='RandomSplit')
    results_dfs.append(newP.desc_df)
    newP = Prediction(main_df, [2015],
            [gpa, 'Initial PGR', 'IsMale', 'IsBlack', 'IsLatino',
                'Academic_Identity','Support_Networks_Family'],
            outcome,
            '2015 base GPA/GR w demo; 2 survey', require=None,remove=None,
            train=None)
    results_dfs.append(newP.desc_df)
    newP = Prediction(main_df, [2015],
            [gpa, 'Initial PGR', 'IsMale', 'IsBlack', 'IsLatino',
                'Academic_Identity','Support_Networks_Family',
                'Organization_Time_Management','Growth_Mindset_Self_Efficacy'],
            outcome,
            '2015 base GPA/GR w demo; 4 survey', require=None,remove=None,
            train=None)
    results_dfs.append(newP.desc_df)
    newP = Prediction(main_df, [2015],
            [gpa, 'Initial PGR', 'IsMale', 'IsBlack', 'IsLatino',
             'Self_Concept', 'Growth_Mindset_Self_Efficacy',
             'Self_Regulation', 'Support_Networks_School',
             'Intrinsic_Motivation', 'Academic_Delay_of_Gratification',
             'Support_Networks_Family', 'HS_Preparation',
             'Performance_Avoidance', 'Academic_Identity',
             'Organization_Time_Management'
             ],
            outcome,
            '2015 base GPA/GR w demo; survey vars', require=None,remove=None,
            train=None)
    results_dfs.append(newP.desc_df)




    # combine everything for output
    full_stats = pd.concat(results_dfs)
    full_coefs = pd.concat(coefs_dfs)
    full_stats.to_csv('outcomes_details.csv')
    full_coefs.to_csv('coefs_details.csv')




if __name__ == '__main__':
    main(
            'inputs/Senior_Survey_Data.csv',
            'inputs/Senior_Survey_Key.csv',
            'inputs/Persistence_Data.csv',
            'inputs/Trials_Specs.csv',
            gpa='WGPA',
            outcome='Retention3'
            )
