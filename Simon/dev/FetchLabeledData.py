# -*- coding: utf-8 -*-
"""
Created on Fri May 26 19:03:35 2017

@author: azunre
"""
import pandas as pd
import azure_utils.client as client
import graphutils.printSample as printSample
import graphutils.getConnection as gc
import graphutils.insertColumnDatasetJoin as insert
import pandas
import sys
import random
import pyodbc
import timeit
import csv
from LengthStandardizer import *
import traceback

from collections import Counter

# fetch and display info on all labels and how many there are...
def PrintLabeledDataInfo(cursor):
    
    # all labeled data in data-lake    
#    string = "Select cl.name, count(1) from columnlabels cl\ninner join columns_columnlabels" \
#        + " ccl on ccl.columnlabel_id=cl.id\ngroup by cl.name"
    # hand-labeled data in data-lake
    string = "Select cl.name, count(1) from columnlabels cl\ninner join columns_columnlabels" \
        + " ccl on ccl.columnlabel_id=cl.id\nwhere cl.labelsource_id=6\ngroup by cl.name, cl.id"
    cursor.execute(string)
    print("The available summary of data/labels is:")
    print(cursor.fetchall())
    
    return
    
# return a list of all the files, and columns within those files 
# which have that particular label
def FetchLabeledDataSummary(cursor, LABEL, DEBUG):
    # all labeled data in data-lake ->
#    string = "Select d.name, c.name from datasets d\n" \
#            + "inner join columns c on c.dataset_id=d.id\n" \
#            + "inner join columns_columnlabels c_cl on c.id=c_cl.column_id\n" \
#            + "inner join columnlabels cl on c_cl.columnlabel_id=cl.id\n" \
#            + "where cl.name=?"
    # hand-labeled data in data-lake ->
    string = "Select d.name, c.name, c.id from datasets d\n" \
            + "inner join columns c on c.dataset_id=d.id\n" \
            + "inner join columns_columnlabels c_cl on c.id=c_cl.column_id\n" \
            + "inner join columnlabels cl on c_cl.columnlabel_id=cl.id\n" \
            + "where cl.name=? and cl.labelsource_id=6"
    # mutually exclusive, hand-labeled ->
#    string = "Select d.name, c.name from columns c\n" \
#            + "inner join datasets d on c.dataset_id=d.id\n" \
#            + "inner join columns_columnlabels c_cl on c.id=c_cl.column_id\n" \
#            + "inner join columnlabels cl on cl.id=c_cl.columnlabel_id\n" \
#            + "left join (select c2.id as colid from columns c2\n" \
#                          + "inner join columns_columnlabels c_cl2 on c2.id=c_cl2.column_id\n" \
#                          + "inner join columnlabels cl2 on cl2.id=c_cl2.columnlabel_id\n" \
#                          + "where cl2.labelsource_id=6 and not cl2.name=?) b on c.id=b.colid\n" \
#            + "where cl.labelsource_id=6 and cl.name=? and b.colid is null"
#
#    cursor.execute(string, LABEL, LABEL)

    cursor.execute(string, LABEL)
    datasets = cursor.fetchall()
    
    if DEBUG:
        print("AVAILABLE FILES, COLUMNS AND LABELS =>")
        print(datasets)
    
    return datasets

# This function was meant to cycle through some delimiters and encodings, will 
# return to this if we find appreciable encoding errors in the future
def get_csv_df(path,limit,offset,adl):
    df = None
    finished = False
    last_retrieved_record = None

    if offset is None:
        offset = 0

    delimiter = ' '
    

    for encoding in ['', 'ascii', 'utf-8', 'utf-16','ISO-8859-1']:
        if df is not None:
            continue
    
        try:

            with adl.open(path) as f:
                df = pd.read_csv(f, nrows=limit, encoding = encoding)

        except Exception:
            continue

    if df is not None:
        if len(df) < limit:
            finished = True
        last_retrieved_record = len(df) - 1 + offset

    return df,finished,last_retrieved_record
    
# actually get the required data for prediction   
def FetchLabeledDataColumns(datasets, max_cells, cursor, adl, DEBUG, LARGE):
    maxlen = 20
    with adl.open(datasets[0][0], blocksize=2**20) as f:
        try:
            frame = pandas.read_csv(f,nrows=max_cells, encoding = "ISO-8859-1",dtype=str)
            out = DataLengthColumnStandardizerRaw(frame.ix[:,datasets[0][1]], max_cells)
            unique_column_id = [datasets[0][2],]
        except:
            print("failed reading " + datasets[0][0])
            if DEBUG:
                raise        
    
    if DEBUG:
        print("OUT, incremental ->")
        print(out)
        print(out.shape)
    for i in np.arange(1,len(datasets)):
        file = datasets[i][0]
        column = datasets[i][1]
                 
        with adl.open(file, blocksize=2**20) as f:
            try_count=0
#            while(try_count<5):
            try:
                frame = pandas.read_csv(f,nrows=max_cells,encoding = "ISO-8859-1",dtype=str)
                if DEBUG:
                    print("starting a new frame...")
                    print(i)
                    print(frame)
                    print(file)
                    print(column)
                if(LARGE):
                    col_data = frame.ix[:,column].str[-maxlen:] #do this for large text files
                else:
                    col_data = frame.ix[:,column]
                out_tmp_1 = DataLengthColumnStandardizerRaw(col_data, max_cells)
                out_tmp_2 = out
                out = np.concatenate((out_tmp_2,out_tmp_1), axis=1)
                unique_column_id.append(datasets[i][2])
            except:
                print("failed reading " + file)
                traceback.print_exc()
                if DEBUG:
                    raise
    
    return out, unique_column_id
    
# now actually get the required data for prediction   
def FetchLabeledDataFromDatabase(max_cells, cursor, adl, DEBUG):    
    try:
        # first, provide summary information...
        PrintLabeledDataInfo(cursor)
        start_time = timeit.default_timer()
        # fetch fixed list of labels
        with open('Categories_base.txt','r') as f:
            Categories = f.read().splitlines()
            
        # go through list of labels and fetch data, while building header
        out_array_header = [ [Categories[0]], ]
        out_array=None
        unique_column_id_array=None
        for category in Categories:
            # LABEL = input("Which label to focus on this time? Please specify: ")
            print("DEBUG::Fetching %s category from database"%category)
            datasets = FetchLabeledDataSummary(cursor, category, DEBUG)
            if(category=='text'):
                out,unique_column_id = FetchLabeledDataColumns(datasets, max_cells, cursor, adl, DEBUG, True)
            else:
                out,unique_column_id = FetchLabeledDataColumns(datasets, max_cells, cursor, adl, DEBUG, False)
            if(np.any(out_array==None)):
                out_array = out
            else:
                out_array = np.column_stack((out_array,out))
            if(unique_column_id_array==None):
                unique_column_id_array = unique_column_id
            else:
                unique_column_id_array.extend(unique_column_id)
            nx,ny = out.shape
            print("DEBUG::%d cells by %d samples fetched..."%(nx,ny))
            #print("DEBUG::unique_column_id_array length is:")
            #print(len(unique_column_id_array))
            if(category==Categories[0]):
                for i in np.arange(ny-1):
                    out_array_header.append(list([category]))
            else:
                for i in np.arange(ny):
                    out_array_header.append(list([category]))
                        
        
        # Now, postprocess fetched data to collapse repeated multilabeled columns
        if DEBUG:
            print("DEBUG::unique_column_id_array::")
            print(len(unique_column_id_array))
            print(unique_column_id_array)
            print("DEBUG::(number of) unique elements::")
            print(len(set(unique_column_id_array)))
        
        
        count = Counter(unique_column_id_array)
        if DEBUG:
            print("DEBUG::count:")
            print(count)
        duplicates = [el for el,ct in count.items() if ct>1]
        unique_column_id_np_array = np.asarray(unique_column_id_array)
        if DEBUG:
            print("DEBUG::duplicates::")
            print(duplicates)
            print(len(duplicates))
            print("DEBUG::out_array size before starting processing is:")
            print(out_array.shape)
            print("DEBUG::out_array_header length now is:")
            print(len(out_array_header))
            print("DEBUG::unique_column_id_np_array shape now is:")
            print(len(unique_column_id_np_array))
        for k in np.arange(len(duplicates)):
            indices_mask = unique_column_id_np_array==duplicates[k]
            indices_list = [i for i,x in enumerate(indices_mask) if x]
            if DEBUG:
                print("DEBUG::indices_list,k::")
                print(indices_list)
                print(k)
                print("DEBUG::out_array size now is:")
                print(out_array.shape)
                print("DEBUG::out_array_header length now is:")
                print(len(out_array_header))
                print("DEBUG::unique_column_id_np_array shape now is:")
                print(len(unique_column_id_np_array))
            out_array_header[indices_list[0]].extend([out_array_header[j][0] for j in indices_list[1:]])
            for i in sorted(indices_list[1:], reverse=True):
                del out_array_header[i]
            #del out_array_header[np.asarray(indices_list[1:])]
            out_array = np.delete(out_array, np.asarray(indices_list[1:]),axis=1)
            unique_column_id_np_array = np.delete(unique_column_id_np_array, np.asarray(indices_list[1:]))
        
        if DEBUG:
            print(out_array)
            print("OUT, final (array) form shape->")
            print(out.shape)
    except:
        print("SQL queries failed, please investigate why...")
        print(sys.exc_info())
        raise
             
    print("DONE w/ SQL queries...!!!")
    
    elapsed_time = timeit.default_timer()-start_time
    print("Time elapsed for DB calls (getting data from db+multilabel postprocess) is: %.2f sec" % elapsed_time)
    
    return out_array,out_array_header
