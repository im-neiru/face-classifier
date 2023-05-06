import csv
import os
#age_set = {'more than 70', '40-49', '3-9', '0-2',
#           '10-19', '50-59', '20-29', '60-69', '30-39'}
#race_set = {'Black', 'Latino_Hispanic', 'Middle Eastern',
#            'Indian', 'Southeast Asian', 'White', 'East Asian'}

dataset = 'dataset'

def race_vector(race):
    if race == 'Black':
        return [1, 0, 0, 0, 0, 0, 0]
    elif race == 'Latino_Hispanic':
        return [0, 1, 0, 0, 0, 0, 0]
    elif race == 'Middle Eastern':
        return [0, 0, 1, 0, 0, 0, 0]
    elif race == 'Indian':
        return [0, 0, 0, 1, 0, 0, 0]
    elif race == 'Southeast Asian':
        return [0, 0, 0, 0, 1, 0, 0]
    elif race == 'White':
        return [0, 0, 0, 0, 0, 1, 0]
    elif race == 'East Asian':
        return [0, 0, 0, 0, 0, 0, 1]
    else:
        return [0, 0, 0, 0, 0, 0, 0]


def age_vector(age):
    if age == 'more than 70':
        return [1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif age == '40-49':
        return [0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif age == '3-9':
        return [0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif age == '0-2':
        return [0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif age == '10-19':
        return [0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif age == '50-59':
        return [0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif age == '20-29':
        return [0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif age == '60-69':
        return [0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif age == '30-39':
        return [0, 0, 0, 0, 0, 0, 0, 0, 1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]


def gender_vector(age):
    if age == 'Female':
        return [1, 0]
    elif age == 'Male':
        return [0, 1]
    else:
        return [0, 0]

writer = csv.writer(open(os.path.join(dataset, 'train_vector.csv'), 'a'))
with open(os.path.join(dataset, 'train.csv')) as input_file:
    for row in csv.reader(input_file):
        new_row = [row[0]]
        
        new_row.extend(gender_vector(row[2]));
        new_row.extend(race_vector(row[3]));
        new_row.extend(age_vector(row[1]));
        writer.writerow(new_row)

writer = csv.writer(open(os.path.join(dataset, 'val_vector.csv'), 'a'))
with open(os.path.join(dataset, 'val.csv')) as input_file:
    for row in csv.reader(input_file):
        new_row = [row[0]]
        
        new_row.extend(gender_vector(row[2]));
        new_row.extend(race_vector(row[3]));
        new_row.extend(age_vector(row[1]));
        writer.writerow(new_row)
        
