import csv
import os
import re

import numpy as np
import pandas as pd

from functools import reduce
from modules.classes.item_id_parser import ItemIDParser


def map_dict(elem, dictionary):
    if elem in dictionary:
        return dictionary[elem]
    else:
        return np.nan


def create_time_feature(series, window_size=24):
    time_strings = series.astype('str').str.split(' ')
    return time_strings.apply(lambda x: f'{x[0]}_' + str(int(int(x[1].split(':')[0]) / window_size)))


class MimicParser(object):
    """
    This class processes a MIMIC database into a single file
    """

    def __init__(self, mimic_folder_path, folder, file_name, id_column, label_column, mimic_version=4, window_size=24):

        if mimic_version == 4:
            id_column = id_column.lower()
            label_column = label_column.lower()
        elif mimic_version == 3:
            id_column = id_column.upper()
            label_column = label_column.upper()
        else:
            raise Exception(f"Unsupported Mimic Version: {mimic_version}")

        self.mimic_version = mimic_version
        self.mimic_folder_path = mimic_folder_path
        self.output_folder = f'{mimic_folder_path}/{folder}'
        self.base_file_name = file_name
        self.window_size = window_size
        self.standard_path = f'{self.output_folder}/{self.base_file_name}'
        self.pid = ItemIDParser(mimic_folder_path, id_column, label_column)

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Output paths for the files produced by methods below
        self.rt_path = self.standard_path + '_0_reduced'
        self.cdb_path = self.standard_path + '_1_24hrs_blocks'
        self.aac_path = self.standard_path + '_2_p_admissions'
        self.apc_path = self.standard_path + '_3_p_patients'
        self.ap_path = self.standard_path + '_4_p_scripts'
        self.aii_path = self.standard_path + '_5_p_icds'
        self.an_path = self.standard_path + '_6_p_notes'

    def reduce_total(self):
        """
        This will filter out rows from CHARTEVENTS.csv that are not feature relevant
        """
        feature_relevant_columns = ['subject_id', 'hadm_id', 'itemid', 'charttime', 'value', 'valuenum']
        if self.mimic_version == 3:
            feature_relevant_columns += ['icustay_id']
        elif self.mimic_version == 4:
            feature_relevant_columns += ['stay_id']
        feature_dict = self.pid.get_feature_dictionary()
        chunksize = 10000000

        mode = 'w'
        header = True
        print('Start processing df')
        for i, df_chunk in enumerate(
                pd.read_csv(self.mimic_folder_path + '/CHARTEVENTS.csv', iterator=True, chunksize=chunksize)):
            print(f'\rChunk number: {i}', end='')
            df_chunk.columns = df_chunk.columns.str.lower()
            df = df_chunk[df_chunk['itemid'].isin(reduce(lambda x, y: x.union(y), feature_dict.values()))]
            df = df.dropna(inplace=False, axis=0, subset=feature_relevant_columns)
            if i == 1:
                mode = 'a'
                header = None
            df.to_csv(self.rt_path + '.csv', index=False, columns=feature_relevant_columns,
                      header=header, mode=mode)

        print(f"\r Finished reducing chart events")

    def create_day_blocks(self):
        """
        Create the time feature as well as std, min and max
        """

        reversed_feature_dict = self.pid.get_reversed_feature_dictionary()
        df = pd.read_csv(self.rt_path + '.csv')
        print("Loaded df")

        # create time feature
        df['chartday'] = create_time_feature(df['charttime'], self.window_size)

        print("New feature chartday")
        df['hadmid_day'] = df['hadm_id'].astype('str') + '_' + df['chartday']
        print("New feature hadmid_day")
        df['features'] = df['itemid'].apply(lambda x: reversed_feature_dict[x])
        print("New feature features")

        hadm_dict = dict(zip(df['hadmid_day'], df['subject_id']))
        print("Start statistics")
        df2 = pd.pivot_table(df, index='hadmid_day', columns='features',
                             values='valuenum', fill_value=np.nan)
        df3 = pd.pivot_table(df, index='hadmid_day', columns='features',
                             values='valuenum', aggfunc=lambda x: np.std(x), fill_value=0)
        df3.columns = ["{0}_std".format(i) for i in list(df2.columns)]
        print("std finished")
        df4 = pd.pivot_table(df, index='hadmid_day', columns='features',
                             values='valuenum', aggfunc=np.amin, fill_value=np.nan)
        df4.columns = ["{0}_min".format(i) for i in list(df2.columns)]
        print("min finished")
        df5 = pd.pivot_table(df, index='hadmid_day', columns='features',
                             values='valuenum', aggfunc=np.amax, fill_value=np.nan)
        df5.columns = ["{0}_max".format(i) for i in list(df2.columns)]
        print("max finished")
        df2 = pd.concat([df2, df3, df4, df5], axis=1)
        df2.columns = df2.columns.str.lower()

        if 'tobacco' in df2.columns:
            df2['tobacco'].apply(lambda x: np.around(x))
            del df2['tobacco_std']
            del df2['tobacco_min']
            del df2['tobacco_max']

        del df2['daily weight_std']
        del df2['daily weight_min']
        del df2['daily weight_max']

        rel_columns = [i for i in list(df2.columns) if '_' not in i]

        for col in rel_columns:
            if len(np.unique(df2[col])[np.isfinite(np.unique(df2[col]))]) <= 2:
                print(col)
                del df2[col + '_std']
                del df2[col + '_min']
                del df2[col + '_max']

        for i in list(df2.columns):
            df2[i][df2[i] > df2[i].quantile(.95)] = df2[i].median()
            df2[i].fillna(df2[i].median(), inplace=True)

        df2['hadmid_day'] = df2.index

        if 'pt' in df2.columns:
            df2['inr'] = df2['inr'] + df2['pt']
            df2['inr_std'] = df2['inr_std'] + df2['pt_std']
            df2['inr_min'] = df2['inr_min'] + df2['pt_min']
            df2['inr_max'] = df2['inr_max'] + df2['pt_max']
            del df2['pt']
            del df2['pt_std']
            del df2['pt_min']
            del df2['pt_max']

        df2.dropna(thresh=int(0.75 * len(df2.columns)), axis=0, inplace=True)
        df2.to_csv(self.cdb_path + '.csv', index=False)
        print("Created Dayblocks")
        return hadm_dict

    def add_admissions_columns(self):
        """
        Adds a field whether a person is black (TODO why???)
        """

        df = pd.read_csv(f'{self.mimic_folder_path}/ADMISSIONS.csv')
        df.columns = df.columns.str.lower()
        ethn_dict = dict(zip(df['hadm_id'], df['ethnicity']))
        admittime_dict = dict(zip(df['hadm_id'], df['admittime']))

        file_name = self.cdb_path + '.csv'
        df_shard = pd.read_csv(file_name)
        df_shard['hadm_id'] = df_shard['hadmid_day'].str.split('_').apply(lambda x: x[0])
        df_shard['hadm_id'] = df_shard['hadm_id'].astype('int')
        df_shard['ethnicity'] = df_shard['hadm_id'].apply(lambda x: map_dict(x, ethn_dict))
        black_condition = df_shard['ethnicity'].str.contains('.*black.*', flags=re.IGNORECASE)
        df_shard['black'] = 0
        df_shard['black'][black_condition] = 1
        del df_shard['ethnicity']
        df_shard['admittime'] = df_shard['hadm_id'].apply(lambda x: map_dict(x, admittime_dict))
        df_shard.to_csv(self.aac_path + '.csv', index=False)
        print("Created 24h blocks")

    def add_patient_columns(self, hadm_dict):
        """
        Add gender columns
        @param hadm_dict:
        """
        df = pd.read_csv(self.mimic_folder_path + '/PATIENTS.csv')
        df.columns = df.columns.str.lower()

        gender_dict = dict(zip(df['subject_id'], df['gender']))
        df_shard = pd.read_csv(self.aac_path + '.csv')
        df_shard['subject_id'] = df_shard['hadmid_day'].apply(lambda x: map_dict(x, hadm_dict))
        df_shard['admityear'] = df_shard['admittime'].str.split('-').apply(lambda x: x[0]).astype('int')
        df_shard['gender'] = df_shard['subject_id'].apply(lambda x: map_dict(x, gender_dict))

        if 'dob' in df.columns:
            dob_dict = dict(zip(df['subject_id'], df['dob']))
            df_shard['dob'] = df_shard['subject_id'].apply(lambda x: map_dict(x, dob_dict))
            df_shard['yob'] = df_shard['dob'].str.split('-').apply(lambda x: x[0]).astype('int')
            # Date of birth replaced by anchor_age
            df_shard['age'] = df_shard['admityear'].subtract(df_shard['yob'])

        gender_dummied = pd.get_dummies(df_shard['gender'], drop_first=True)
        gender_dummied.rename(columns={'M': 'Male', 'F': 'Female'})
        columns = list(df_shard.columns)
        columns.remove('gender')
        df_shard = pd.concat([df_shard[columns], gender_dummied], axis=1)
        df_shard.to_csv(self.apc_path + '.csv', index=False)
        print("Created Plus Admission")

    def clean_prescriptions(self):
        """
        Only keep relevant prescriptions
        """
        prescriptions = self.pid.get_prescriptions(self.mimic_version)
        prescriptions.columns = prescriptions.columns.str.lower()
        prescriptions.drop_duplicates(inplace=True)
        prescriptions['drug_feature'] = np.nan
        print(f"{len(self.pid.script_features_names)}")
        count = 1
        for feature, pattern in zip(self.pid.script_features_names, self.pid.script_patterns):
            print(f"\r{count}-{feature}", end=" ")
            condition = prescriptions['drug'].str.contains(pattern, flags=re.IGNORECASE)
            prescriptions['drug_feature'][condition] = feature
            count += 1
        prescriptions.dropna(how='any', axis=0, inplace=True, subset=['drug_feature'])
        prescriptions.to_csv(self.output_folder + '/PRESCRIPTIONS_reduced.csv', index=False)
        print("Cleaned Prescriptions")

    def add_prescriptions(self):
        """
        Add drug prescriptions
        """
        file_name = 'PRESCRIPTIONS_reduced'
        if self.mimic_version == 3:
            max_index = 3  # Go until hadm_id
            startdate_index = 3
            enddate_index = 4
        elif self.mimic_version == 4:
            max_index = 2  # Go until hadm_id
            startdate_index = 2
            enddate_index = 3
        with open(f'{self.output_folder}/{file_name}.csv', 'r') as f:
            print(f"Number of rows: {sum(1 for _ in csv.reader(f))}")
        with open(f'{self.output_folder}/{file_name}.csv', 'r') as f:
            csvreader = csv.reader(f)
            with open(f'{self.output_folder}/{file_name}_byday.csv', 'w') as g:
                csvwriter = csv.writer(g)
                first_line = csvreader.__next__()
                print(first_line[0:max_index] + ['chartday'] + [first_line[-1]])
                csvwriter.writerow(first_line[0:max_index] + ['chartday'] + [first_line[-1]])
                count = 1
                for row in csvreader:
                    dates = pd.date_range(row[startdate_index], row[enddate_index],
                                          freq=f'{self.window_size}h').strftime('%Y-%m-%d %H:%M:%S')
                    dates = create_time_feature(pd.Series(dates), self.window_size)
                    for date in dates:
                        csvwriter.writerow(row[0:max_index] + [date] + [row[-1]])
                    print(f"\r{count}", end="")
                    count += 1
        print('\rFinished writing PRESCRIPTIONS_reduced_byday.csv')

        df = pd.read_csv(self.output_folder + '/PRESCRIPTIONS_reduced_byday.csv')
        df['chartday'] = df['chartday'].str.split(' ').apply(lambda x: x[0])
        df['hadmid_day'] = df['hadm_id'].astype('str') + '_' + df['chartday']
        df['value'] = 1

        cols = ['hadmid_day', 'drug_feature', 'value']
        df = df[cols]

        df_pivot = pd.pivot_table(df, index='hadmid_day', columns='drug_feature', values='value', fill_value=0,
                                  aggfunc=np.amax)
        df_pivot.reset_index(inplace=True)

        df_file = pd.read_csv(self.apc_path + '.csv')
        df_merged = pd.merge(df_file, df_pivot, on='hadmid_day', how='outer')

        del df_merged['hadm_id']
        df_merged['hadm_id'] = df_merged['hadmid_day'].str.split('_').apply(lambda x: x[0])
        df_merged.fillna(0, inplace=True)

        df_merged['dextrose'] = df_merged['dextrose'] + df_merged['D5W']
        del df_merged['D5W']

        df_merged.to_csv(self.ap_path + '.csv', index=False)
        print("Added Prescriptions")

    def add_icd_infect(self):
        """
        Add icd infect column
        """
        df_icd = pd.read_csv(self.mimic_folder_path + '/PROCEDURES_ICD.csv')
        df_icd.columns = df_icd.columns.str.lower()
        df_micro = pd.read_csv(self.mimic_folder_path + '/MICROBIOLOGYEVENTS.csv')
        df_micro.columns = df_micro.columns.str.lower()
        suspect_hadmid = set(pd.unique(df_micro['hadm_id']).tolist())

        if self.mimic_version == 3:
            df_icd_ckd = df_icd[df_icd['icd9_code'] == 585]
        elif self.mimic_version == 4:
            df_icd_ckd = df_icd[(df_icd['icd_code'] == 585) | (df_icd['icd_code'] == 'N18.9')]

        ckd = set(df_icd_ckd['hadm_id'].values.tolist())

        df = pd.read_csv(self.ap_path + '.csv')
        df['ckd'] = df['hadm_id'].apply(lambda x: 1 if x in ckd else 0)
        df['infection'] = df['hadm_id'].apply(lambda x: 1 if x in suspect_hadmid else 0)
        df.to_csv(self.aii_path + '.csv', index=False)
        print("Added ICD Infections")

    def add_notes(self):
        """
        Add features for note events
        """
        df = pd.read_csv(self.mimic_folder_path + '/NOTEEVENTS.csv')
        df.columns = df.columns.str.lower()
        df_rad_notes = df[['text', 'hadm_id']][df['category'] == 'Radiology']
        cta = df_rad_notes['text'].str.contains('CTA', flags=re.IGNORECASE)
        ct_angiogram_bool_array = df_rad_notes['text'].str.contains('CT angiogram', flags=re.IGNORECASE)
        chest_angiogram_bool_array = df_rad_notes['text'].str.contains('chest angiogram', flags=re.IGNORECASE)
        cta_hadm_ids = np.unique(df_rad_notes['hadm_id'][cta].dropna())
        ct_angiogram_hadm_ids = np.unique(df_rad_notes['hadm_id'][ct_angiogram_bool_array].dropna())
        chest_angiogram_hadm_ids = np.unique(df_rad_notes['hadm_id'][chest_angiogram_bool_array].dropna())
        hadm_id_set = set(cta_hadm_ids.tolist())
        hadm_id_set.update(ct_angiogram_hadm_ids)
        print(len(hadm_id_set))
        hadm_id_set.update(chest_angiogram_hadm_ids)
        print(len(hadm_id_set))

        df_shard = pd.read_csv(self.aii_path + '.csv')
        df_shard['ct_angio'] = df_shard['hadm_id'].apply(lambda x: 1 if x in hadm_id_set else 0)
        df_shard.to_csv(self.an_path + '.csv', index=False)
        print("Added Notes")

    def perform_full_parsing(self):
        """
        Call all methods of self to perform the full pipeline on a mimic db
        """
        self.reduce_total()
        hadm_dict = self.create_day_blocks()
        self.add_admissions_columns()
        self.add_patient_columns(hadm_dict)
        self.clean_prescriptions()
        self.add_prescriptions()
        self.add_icd_infect()
        if self.mimic_version == 3:
            self.add_notes()
            file_path = self.an_path + '.csv'
        else:
            file_path = self.aii_path + '.csv'
        print(f"Finished Parsing MIMIC {self.mimic_version}\nFinal file can be found under:\n{file_path}")
