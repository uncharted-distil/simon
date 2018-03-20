# -*- coding: utf-8 -*-
"""
Created on Mon May 15 18:24:27 2017

@author: azunre
"""

import pyodbc

def getConnection():
    server ='metadatastore.database.windows.net' #'layerspoc.database.windows.net'
    database ='int-A' #'layered-graph-poc'
    username='nk-admin'  #'david'
    password='NCC-1701-D'


    driver= '{ODBC Driver 13 for SQL Server}'
    return pyodbc.connect('DRIVER='+driver+';PORT=1433;SERVER='+server+';PORT=1443;DATABASE='+database+';UID='+username+';PWD='+ password)