# -*- coding: utf-8 -*-

from dateutil.parser import parse
#from penny.geo_lookup import get_places_by_type
#from address import AddressParser
from email.utils import parseaddr
from .utils import prep_value, strip_non_ascii
# import phonenumbers
import re
import time
import datetime
import os

cur_dir = os.path.dirname(os.path.realpath(__file__))


DELIMITERS = [',','|','/']


def is_a_nothing(value, header=None):
    value = prep_value(value).lower()

    if not value:
        return True
    
    if value in ['none','nan','null','n/a']:
        return True
    
    return False


def is_a_time(value, header=None):
    value = prep_value(value).lower()

    try:
        time.strptime(value, '%H:%M')
        return True
    except ValueError:
        try:
            time.strptime(value, '%H:%M %p')
            return True
        except ValueError:
            try:
                time.strptime(value, '%H:%M:%S')
                return True
            except ValueError:
                return False


def is_a_date(value, header=None):
    """
    Dateutil recognizes some letter combinations as dates, which is almost 
    always something that isn't really a date (like a US state abbreviation)
    """
    value = prep_value(value)

    if len(value) < 4:
        return False

    if value.startswith('-'):
        return False

    if not any(char.isdigit() for char in value):
        return False

    if is_a_time(value, header=header):
        return False

    """
    This is iffy. Obviously it's totally possible to have infinitely 
    precise measurements, but we're going to guess that if there are 
    more numbers to the right of the decimal point than the left, we 
    are probably dealing with a coordinate (or something)
    """
    if '.' in value:
        pieces = value.split('.')
        if len(pieces[1]) > len(pieces[0]):
            return False
        elif len(pieces) == 2 and len(pieces[0]) < 9:
            return False

    now = datetime.datetime.now()
    
    try:
        d = parse(value)
        # Hard to imagine a dataset only in the future
        if d.year < now.year + 5 and d.year > now.year - 300:
            return True
        return False
    except Exception as e:
        try:
            d = datetime.datetime.fromtimestamp(int(value))
            if d.year < now.year + 5 and d.year > now.year - 5:
                return True
        except:
            return False

        return False



def is_a_currency(value, header=None):
    value = prep_value(value)

    if value.startswith((u'$',u'£',u'€')):
        return True

    return False


def is_a_percent(value, header=None):
    value = prep_value(value)
    
    if value.endswith('%'):
        return True

    return False


def is_a_number(value, header=None):
    value = prep_value(value)
    pieces = value.split(',')
    if len(pieces) > 1 and not any([len(piece) == 3 for piece in pieces]):
        return False

    value = value.replace(',','')

    if header:
        header = prep_value(header).lower()

    if header and header in ['zip', 'zipcode', 'postal code']:
        return False

    if is_a_percent(value) or is_a_currency(value):
        return True

    if "." in value:
        try:
            v = float(value)
            return True
        except:
            return False
    else:
        try:
            int(value)
            return True
        except:
            return False


def is_a_text(value, header=None):
    value = prep_value(value).strip()

    return len(value) > 90


def is_a_label(value, header=None):
    value = prep_value(value)

    if not value or value == "":
        return False

    if is_a_number(value):
        return False

    if len(value) > 40:
        return False

    if '.' in value:
        return False

    if len(value.split(' ')) > 5:
        return False

    possible_matches = [d for d in DELIMITERS if d in value]
    
    # more than one delimeter is probably a sentence
    if len(possible_matches) > 1:
        return False

    return True


def is_a_latitude(value, header=None):
    value = prep_value(value)

    if header:
        header = unicode(header).lower().strip()

    if not is_a_number(value):
        return False

    if not abs(float(value)) <= 90:
        return False

    # so we know we have a value that is between -90 and 90
    header_names = ['lat', 'lt']
    if header and any([h in header for h in header_names]):
        return True

    return False


def is_a_longitude(value, header=None):
    value = prep_value(value)

    if header:
        header = unicode(header).lower().strip()

    if not is_a_number(value):
        return False

    if not abs(float(value)) <= 180:
        return False

    # so we know we have a value that is between -180 and 180
    header_names = ['lon', 'lng']
    if header and any([h in header for h in header_names]):
        return True

    return False


def is_a_coord_pair(value, header=None, pos=None):
    value = prep_value(value)

    disallowed = ['(',')','{','}','[',']']

    possible_matches = [d for d in DELIMITERS if d in value]
    
    # if more than one of these is present or none of them, than this isn't 
    # a pair
    if len(possible_matches) != 1:
        return False

    # Get rid of weird shit people put in coord pair columns
    for d in disallowed:
        value = value.replace(d,'')

    delimiter = possible_matches[0]
    possible_cords = value.split(delimiter)
    
    if len(possible_cords) != 2:
        return False

    # All parts have to be floats or ints
    if not all([is_a_number(x) for x in possible_cords]):
        return False

    # max abs lat is 90, max abs lng is 180
    if any([abs(float(x)) > 180 for x in possible_cords]):
        return False

    if all([abs(float(x)) > 90 for x in possible_cords]):
        return False

    # We have a pair of numbers, each of which is less than 180 and one of 
    # which is less than 90
    return True
    

def is_a_place(value, place_type, header=None):
    value = prep_value(value)

    if not is_a_label(value):
        return False

    # If your country's name is longer than 40 characters, you're doing 
    # something wrong.
    if len(value) > 40:
        return False

    non_addrs = ['|','/','?','!','@','$','%']
    if len([na for na in non_addrs if na in value]) > 0:
        return False

    if header:
        header = header.lower().strip()

    if header and header in place_type:
        return True

    if len(get_places_by_type(value, place_type)) > 0:
        return True

    if place_type in ['region'] and len(value) < 4 and \
        len(get_places_by_type(value, place_type + '_iso_code')) > 0:
        return True

    return False


def is_a_city(value, header=None, pos=None):
    return is_a_place(value, 'city', header=header)


def is_a_region(value, header=None, pos=None):
    return is_a_place(value, 'region', header=header)


def is_a_country(value, header=None, pos=None):
    return is_a_place(value, 'country', header=header)


state_list = [l.strip().lower() for l in open(cur_dir + '/data/us_states.txt').readlines()]
def is_a_state(value, header=None):
    value = prep_value(value)
    if value.lower() in state_list:
        return True
    
    return False


def is_a_zip(value, header=None, pos=None):
    value = prep_value(value)

    if header:
        header = prep_value(header).lower()

    if header and header in ['zip', 'zipcode', 'postal code']:
        return True

    if value.count('-') == 1 and len(value) == 10:
        primary = value.split('-')[0]
    else:
        primary = value

    try:
        primary = int(primary)
    except:
        return False

    primary = prep_value(primary)
    if len(primary) == 5 and int(primary) > 499:
        return True

    return False


ap = AddressParser()
def address_pieces(value):
    value = strip_non_ascii(prep_value(value))
    
    if is_a_number(value):
        return [], None

    if len(value) > 80:
        return [], None

    address = ap.parse_address(value)

    keys = [
        'house_number', 
        'street', 
        'city',
        'zip',
        'state'
    ]

    return [key for key in keys if getattr(address, key, None)], address


"""Check if a string is a house number + street name. The street part of an 
address. Note that we return false if this is a more complete address. """
def is_a_street(value, header=None, pos=None):
    has,address = address_pieces(value)

    if len(has) == 2 and 'house_number' in has and 'street' in has:
        return not is_a_address(value, header=header)
    
    return False


"""Check to see if this is enough of an address that it could be geocoded. So 
has at least a city + state, or at least street + city"""
def is_a_address(value, header=None, pos=None):
    value = prep_value(value)
    
    has,address = address_pieces(value)
    
    if len(has) >= 2:
        if getattr(address, 'city', None) is not None:
            return True
        
        pieces = value.split(' ')
        if len(pieces) > 2:
            """Sometimes we get an address like 100 Congress Austin TX, so let's 
            take a stab at breaking up the city/state in a way AddressParser 
            might understand"""
            pieces[len(pieces) - 2] = pieces[len(pieces) - 2] + ','

            has,address = address_pieces(' '.join(pieces))
            if len(has) > 2 and getattr(address, 'city', None):
                return True

    return False


def is_a_phone(value, header=None, pos=None):
    value = prep_value(value)

    if len(value) > 20:
        return False

    """Check for international numbers"""
    if value.startswith('+'):
        try:
            phonenumbers.parse(value)
            return True
        except:
            return False

    """Otherwise let's hope it's a US number"""
    reg = re.compile(".*?(\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}).*?", re.S)
    matches = reg.search(value)

    """We're not looking for text fields that contain phone numbers, only fields 
    that are dedicated to phone number"""
    if matches and len(matches.group(1)) == len(value):
        return True


    return False


def is_a_email(value, header=None, pos=None):
    value = prep_value(value)

    possible = parseaddr(value)
    if possible[1] == '':
        return False
    
    e = re.compile(r'[\w\-][\w\-\.]+@[\w\-][\w\-\.]+[a-zA-Z]{1,4}')
    m = e.search(possible[1])
    if not m:
        return False

    if len(m.group(0)) == len(possible[1]):
        return True

    return False


def is_a_url(value, header=None, pos=None):
    value = prep_value(value)

    # blatantly ripped from Django
    regex = re.compile(
        r'(^https?://)?'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    m = regex.search(value)

    if not m:
        return False

    return len(m.group(0)) == len(value)
