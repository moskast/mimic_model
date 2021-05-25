import pandas as pd
import re


class ItemIDParser(object):
    """
    This class builds the dictionaries depending on desired features
    """

    def __init__(self, file_path, id_column, label_column):
        self.root = file_path
        self.id_column = id_column
        self.label_column = label_column

        self.d_items = pd.read_csv(file_path + '/D_ITEMS.csv', usecols=[id_column, label_column])
        self.d_items.dropna(how='any', axis=0, inplace=True)

        self.feature_names = ['RBCs', 'WBCs', 'platelets', 'hemoglobin', 'hemocrit',
                              'atypical lymphocytes', 'bands', 'basophils', 'eosinophils', 'neutrophils',
                              'lymphocytes', 'monocytes', 'polymorphonuclear leukocytes',
                              'temperature (F)', 'heart rate', 'respiratory rate', 'systolic', 'diastolic',
                              'pulse oximetry',
                              'troponin', 'HDL', 'LDL', 'BUN', 'INR', 'PTT', 'PT', 'triglycerides', 'creatinine',
                              'glucose', 'sodium', 'potassium', 'chloride', 'bicarbonate',
                              'blood culture', 'urine culture', 'surface culture', 'sputum' +
                              ' culture', 'wound culture', 'Inspired O2 Fraction', 'central venous pressure',
                              'PEEP Set', 'tidal volume', 'anion gap',
                              'daily weight', 'tobacco', 'diabetes', 'history of CV events']

        features = ['$^RBC(?! waste)', '$.*wbc(?!.*apache)', '$^platelet(?!.*intake)',
                         '$^hemoglobin', '$hematocrit(?!.*Apache)',
                         'Differential-Atyps', 'Differential-Bands', 'Differential-Basos', 'Differential-Eos',
                         'Differential-Neuts', 'Differential-Lymphs', 'Differential-Monos', 'Differential-Polys',
                         'temperature f', 'heart rate', 'respiratory rate', 'systolic', 'diastolic',
                         'oxymetry(?! )',
                         'troponin', 'HDL', 'LDL', '$^bun(?!.*apache)', 'INR', 'PTT',
                         '$^pt\\b(?!.*splint)(?!.*exp)(?!.*leak)(?!.*family)(?!.*eval)(?!.*insp)(?!.*soft)',
                         'triglyceride', '$.*creatinine(?!.*apache)',
                         '(?<!boost )glucose(?!.*apache).*',
                         '$^sodium(?!.*apache)(?!.*bicarb)(?!.*phos)(?!.*ace)(?!.*chlo)(?!.*citrate)(?!.*bar)(?!.*PO)',
                         '$.*(?<!penicillin G )(?<!urine )potassium(?!.*apache)',
                         '^chloride', 'bicarbonate', 'blood culture', 'urine culture', 'surface culture',
                         'sputum culture', 'wound culture', 'Inspired O2 Fraction', '$Central Venous Pressure(?! )',
                         'PEEP set', 'tidal volume \(set\)', 'anion gap', 'daily weight', 'tobacco', 'diabetes',
                         'CV - past']

        self.patterns = []
        for feature in features:
            if '$' in feature:
                self.patterns.append(feature[1::])
            else:
                self.patterns.append('.*{0}.*'.format(feature))

        self.script_features_names = ['epoetin', 'warfarin', 'heparin', 'enoxaparin', 'fondaparinux',
                                      'asprin', 'ketorolac', 'acetominophen',
                                      'insulin', 'glucagon',
                                      'potassium', 'calcium gluconate',
                                      'fentanyl', 'magensium sulfate',
                                      'D5W', 'dextrose',
                                      'ranitidine', 'ondansetron', 'pantoprazole', 'metoclopramide',
                                      'lisinopril', 'captopril', 'statin',
                                      'hydralazine', 'diltiazem',
                                      'carvedilol', 'metoprolol', 'labetalol', 'atenolol',
                                      'amiodarone', 'digoxin(?!.*fab)',
                                      'clopidogrel', 'nitroprusside', 'nitroglycerin',
                                      'vasopressin', 'hydrochlorothiazide', 'furosemide',
                                      'atropine', 'neostigmine',
                                      'levothyroxine',
                                      'oxycodone', 'hydromorphone', 'fentanyl citrate',
                                      'tacrolimus', 'prednisone',
                                      'phenylephrine', 'norepinephrine',
                                      'haloperidol', 'phenytoin', 'trazodone', 'levetiracetam',
                                      'diazepam', 'clonazepam',
                                      'propofol', 'zolpidem', 'midazolam',
                                      'albuterol', 'ipratropium',
                                      'diphenhydramine',
                                      '0.9% Sodium Chloride',
                                      'phytonadione',
                                      'metronidazole',
                                      'cefazolin', 'cefepime', 'vancomycin', 'levofloxacin',
                                      'cipfloxacin', 'fluconazole',
                                      'meropenem', 'ceftriaxone', 'piperacillin',
                                      'ampicillin-sulbactam', 'nafcillin', 'oxacillin',
                                      'amoxicillin', 'penicillin', 'SMX-TMP']

        script_features = ['epoetin', 'warfarin', 'heparin', 'enoxaparin', 'fondaparinux',
                           'aspirin', 'keterolac', 'acetaminophen',
                           'insulin', 'glucagon',
                           'potassium', 'calcium gluconate',
                           'fentanyl', 'magnesium sulfate',
                           'D5W', 'dextrose',
                           'ranitidine', 'ondansetron', 'pantoprazole', 'metoclopramide',
                           'lisinopril', 'captopril', 'statin',
                           'hydralazine', 'diltiazem',
                           'carvedilol', 'metoprolol', 'labetalol', 'atenolol',
                           'amiodarone', 'digoxin(?!.*fab)',
                           'clopidogrel', 'nitroprusside', 'nitroglycerin',
                           'vasopressin', 'hydrochlorothiazide', 'furosemide',
                           'atropine', 'neostigmine',
                           'levothyroxine',
                           'oxycodone', 'hydromorphone', 'fentanyl citrate',
                           'tacrolimus', 'prednisone',
                           'phenylephrine', 'norepinephrine',
                           'haloperidol', 'phenytoin', 'trazodone', 'levetiracetam',
                           'diazepam', 'clonazepam',
                           'propofol', 'zolpidem', 'midazolam',
                           'albuterol', '^ipratropium',
                           'diphenhydramine(?!.*%)(?!.*cream)(?!.*/)',
                           '^0.9% sodium chloride(?! )',
                           'phytonadione',
                           'metronidazole(?!.*%)(?! desensit)',
                           'cefazolin(?! )', 'cefepime(?! )', 'vancomycin', 'levofloxacin',
                           'cipfloxacin(?!.*ophth)', 'fluconazole(?! desensit)',
                           'meropenem(?! )', 'ceftriaxone(?! desensit)', 'piperacillin',
                           'ampicillin-sulbactam', 'nafcillin', 'oxacillin', 'amoxicillin',
                           'penicillin(?!.*Desen)', 'sulfamethoxazole']

        self.script_patterns = ['.*' + feature + '.*' for feature in script_features]

    def get_prescriptions(self, mimic_version):
        """
        Get prescriptions from .csv
        @param mimic_version: which mimic version to use
        @return: prescriptions extracted from the .csv
        """
        if mimic_version == 3:
            columns = ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'DRUG', 'STARTDATE', 'ENDDATE']
        elif mimic_version == 4:
            columns = ['subject_id', 'hadm_id', 'drug', 'starttime', 'stoptime']
        else:
            raise Exception(f"Unsupported Mimic Version: {mimic_version}")
        prescriptions = pd.read_csv(self.root + '/PRESCRIPTIONS.csv', usecols=columns)
        prescriptions.dropna(how='any', axis=0, inplace=True)
        return prescriptions

    def get_feature_dictionary(self):
        """
        Assemble feature dictionary from feature names
        @return: a dictionary containing the rows in which items for each feature are held
        """
        assert len(self.feature_names) == len(self.patterns)
        feature_dictionary = dict()
        for feature, pattern in zip(self.feature_names, self.patterns):
            condition = self.d_items[self.label_column].str.contains(pattern, flags=re.IGNORECASE)
            dictionary_value = self.d_items[self.id_column].where(condition).dropna().values.astype('int')
            feature_dictionary[feature] = set(dictionary_value)
        return feature_dictionary

    def get_reversed_feature_dictionary(self):
        """
        Swaps keys and values of a dictionary
        @return: reversed feature dictionary
        """
        rev = {}
        for key, value in self.get_feature_dictionary().items():
            for elem in value:
                rev[elem] = key
        return rev
