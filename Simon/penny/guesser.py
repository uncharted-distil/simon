from collections import Counter
from .utils import to_unicode
''' from .value_checks import (is_a_date, is_a_number, is_a_nothing,
    is_a_latitude, is_a_longitude, is_a_coord_pair, is_a_country, is_a_city, 
    is_a_state, is_a_address, is_a_text, is_a_label, is_a_zip, is_a_street, 
    is_a_phone, is_a_url, is_a_email, is_a_time, is_a_currency, is_a_percent) '''

# currently understands
# category
# datetime
# time
# number
# label
# text
# id
# email
# url
# address
# street
# city
# state
# zipcode
# country
# phone
# latitude
# longitude
# coordinate_pair

# coming soon
# name

# ordinal??? -- can obtain from categorical/int info...

from .utils import prep_value

def is_a_nothing(value, header=None):
    value = prep_value(value).lower()

    if not value:
        return True
    
    if value in ['none','nan','null','n/a']:
        return True
    
    return False


def guess(column_values, header=None, for_types=None):
    types = []
    checked_types = []
    threshold = .8

    def should_check(column_type):
        if not for_types or column_type in for_types:
            return True

        return False

    # Check if all values are unique
    if should_check('rowlabel'):
        if len(list(set(column_values))) == len(column_values):
            types.append('rowlabel')

    # Get non-empty values
    values = [v for v in column_values if not is_a_nothing(v)]
    count_not_empty = len(values)
    
    # If the column is empty, we don't need to do anything else
    if count_not_empty == 0:
        types.append('unknown')
        return sorted(list(set(types)))
    
    
    def do_check(test_func):
        passes_test_count = sum([test_func(v, header=header) for v in values])
        return  float(passes_test_count) / count_not_empty > threshold
    
    
    # if the column is long text, we don't need to do anything else
    if should_check('text') and do_check(is_a_text):
        types.append('text')
        return sorted(list(set(types)))


    # If the column is a date, we don't need to do anything else
    if should_check('datetime') and do_check(is_a_date):
        types.append('datetime')
        return sorted(list(set(types)))

    word_counts = Counter(values).items()
    largest_category_ratio = float(max([w[1] for w in word_counts])) / len(values)
    unique_value_ratio = float(len(word_counts)) / len(values)
    dot_ratio = float(sum([u'.' in to_unicode(v) for v in values])) / len(values)
    
    

    # periods are important to check for determining if numeric is a category
    has_dots = dot_ratio > threshold
    is_boolean = len(word_counts) == 2
    
    # See if values repeat often enough that we should count groups
    if should_check('category'):
        if is_boolean or (largest_category_ratio >= .05 and \
            unique_value_ratio < .2 and not has_dots and len(word_counts)<50):
            
            #print(largest_category_ratio)
            #print(unique_value_ratio)
            #print(word_counts)
            
            types.append('category')


    # Check if this is some kind of ID
    id_labels = ['_id', '_ID', '-id', '-ID', 'ID', ' ID', ' id']
    is_id_header = header and any([s in header for s in id_labels])

    
    if ('rowlabel' in types or 'category' in types) and is_id_header:
        return sorted(list(set(types)))

    
    # Check for number stuff
    if not is_id_header and should_check('numeric') and \
        do_check(is_a_number) and not is_boolean:
        
        types.append('numeric')
        
        # more number stuff
        if should_check('currency') and do_check(is_a_currency):
            types.append('currency')

        elif should_check('percent') and do_check(is_a_percent):
            types.append('percent')

        elif should_check('latitude') and do_check(is_a_latitude):
            types.append('latitude')

        elif should_check('longitude') and do_check(is_a_longitude):
            types.append('longitude')

    else:
        # string stuff
        if should_check('email') and do_check(is_a_email):
            types += ['email', 'label']

        elif should_check('url') and do_check(is_a_url):
            types += ['url']

        elif should_check('time') and do_check(is_a_time):
            types.append('time')

        elif should_check('coordinate') and do_check(is_a_coord_pair):
            types.append('coordinate')

        elif should_check('phone') and do_check(is_a_phone):
            types += ['phone']

        elif should_check('zip') and do_check(is_a_zip):
            types += ['zip']

        elif should_check('state') and do_check(is_a_state):
            types += ['state']

        elif should_check('country') and do_check(is_a_country):
            types += ['country']

        elif should_check('city') and do_check(is_a_city):
            types += ['city'] 

        elif should_check('address') and do_check(is_a_address):
            types.append('address')

        elif should_check('street') and do_check(is_a_street):
            types += ['street']

        elif should_check('label') and do_check(is_a_label):
            types += ['label']

    
    if len(types) == 0:
        types.append('unknown')

    return sorted(list(set(types)))
