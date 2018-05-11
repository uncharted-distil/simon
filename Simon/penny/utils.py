from datetime import datetime

def to_unicode(w):
    if isinstance(w, str):
        return w
#    if isinstance(w, basestring):
#        return str(w, errors='ignore')
    else:
        return str(w)

def prep_value(w):
    return to_unicode(w).strip()


def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    if type(string) == datetime:
        return str(string)
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)
