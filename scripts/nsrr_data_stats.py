from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pprint import pprint

import numpy as np
import pandas as pd

project_root = os.path.abspath('..')
sys.path.append(project_root)

from sleeprnn.helpers import reader
from sleeprnn.common import pkeys, constants
from sleeprnn.data.nsrr_ss import SUBDATASETS

if __name__ == '__main__':

    nsrr = reader.load_dataset(constants.NSRR_SS_NAME)

    # Form table of stats
    table = {
        'subdataset': [],
        'subject_id': [],
        'n2_hours': [],
        'age': [],
        'female': []
    }
    for i_sub, subject_id in enumerate(nsrr.all_ids):
        subdataset_id = subject_id[:4]
        subject_data = nsrr.read_subject_data(subject_id)

        n2_hours = (subject_data['hypnogram'] == nsrr.n2_id).sum() * nsrr.original_page_duration / 3600

        table['subdataset'].append(subdataset_id)
        table['subject_id'].append(subject_id)
        table['n2_hours'].append(n2_hours)
        table['age'].append(subject_data['age'])
        table['female'].append(int(subject_data['sex'] == 'f'))
        print("Progress %d / %d" % (i_sub + 1, len(nsrr.all_ids)))

    table = pd.DataFrame.from_dict(table)

    # Now subgroups
    subdatasets = np.unique(table.subdataset)

    for subdataset in subdatasets:
        table_sub = table[table.subdataset == subdataset]
        total_subjects = len(table_sub)
        total_hours = table_sub.n2_hours.values.sum()
        age_min = table_sub.age.values.min()
        age_max = table_sub.age.values.max()
        age_mean = table_sub.age.values.mean()
        female_fraction = table_sub.female.values.mean()
        print("\nSubdataset:", subdataset)
        print("    N = %d" % total_subjects)
        print("    Size (h) = %1.1f" % total_hours)
        print("    Age min-mean-max: %1.1f - %1.1f - %1.1f" % (age_min, age_mean, age_max))
        print("    Sex (female): %1.1f" % (100 * female_fraction))

    # Now entire dataset
    table_sub = table
    total_subjects = len(table_sub)
    total_hours = table_sub.n2_hours.values.sum()
    age_min = table_sub.age.values.min()
    age_max = table_sub.age.values.max()
    age_mean = table_sub.age.values.mean()
    female_fraction = table_sub.female.values.mean()
    print("\nCombination:")
    print("    N = %d" % total_subjects)
    print("    Size (h) = %1.1f" % total_hours)
    print("    Age min-mean-max: %1.1f - %1.1f - %1.1f" % (age_min, age_mean, age_max))
    print("    Sex (female): %1.1f" % (100 * female_fraction))
