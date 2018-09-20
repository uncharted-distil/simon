import pyodbc
from graphutils.genericJoin import * 
from graphutils.getConnection import *

def insertColumnDatasetJoin(column, dataset, cnxn):
	genericJoin("columns", dataset + "_" + column, "datasets", dataset, "columns_datasets", 1000, cnxn)
	genericJoin("columns", dataset + "_" + column, "columnnames", column, "columns_columnnames", 1000, cnxn)

def labelColumn(dataset,column,label,cnxn):
	genericJoin("columns", dataset + "_" + column, "columnlabels", label, "columns_columnlabels", 1000, cnxn)

def updateColumnLabel(dataset,column,label,cnxn):
	cursor = cnxn.cursor()

	# check to see if entry 1 already exists
	cursor.execute("SELECT * FROM columns WHERE name = CAST(? as varchar(max))", dataset + "_" + column)
	row = cursor.fetchone()
	if row:
		cursor.execute("DELETE FROM columns_columnlabels WHERE id_1 = ?",row[0])
	labelColumn(dataset, column, label, cnxn)