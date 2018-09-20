import pyodbc
import operator
import getConnection as gc

def datasetsToColumns(datasets, cnxn):
	cursor = cnxn.cursor()

	columns_dict = dict()

	for dataset in datasets:
		cursor.execute("SELECT c.* from datasets d INNER JOIN columns_datasets cd on cd.id_2=d.id INNER JOIN columns c on cd.id_1=c.id WHERE d.id=?",dataset)

		columns = cursor.fetchall()
		for column in columns:
			if column[0] in columns_dict:
				columns_dict[column[0]] = columns_dict[column[0]] + 1
			else:
				columns_dict[column[0]] = 1

	return columns_dict

def columnsToDatasets(columns, cnxn):
	cursor = cnxn.cursor()

	datasets_dict = dict()

	for column in columns:
		cursor.execute("SELECT d.* from columns c INNER JOIN columns_datasets cd on cd.id_1=c.id INNER JOIN datasets d on d.id=cd.id_2 WHERE c.id=?",column)
		datasets = cursor.fetchall()
		for dataset in datasets:
			if dataset[0] in datasets_dict:
				datasets_dict[dataset[0]] = datasets_dict[dataset[0]] + 1
			else:
				datasets_dict[dataset[0]] = 1

	return datasets_dict

def getColumnNamesFromIds(ids,cnxn):
	cursor = cnxn.cursor()

	names = {}

	for ID in ids:
		cursor.execute("SELECT name FROM columns WHERE id=?",ID[0])
		name = cursor.fetchone()
		names[ID[0]] = name

	return names
def getSimilarColumns(column):
	print("Datasets using " + column)
	cnxn = gc.getConnection()

	# coincident: two columns which are found in the same dataset, or two datasets containig the same column

	# get datasets which contain this column
	datasetsWithColumn = columnsToDatasets([column], cnxn)
	
	# get other columns which appear in the same datasets as "column"
	coincidentColumns = datasetsToColumns(datasetsWithColumn.keys(), cnxn)
	# Can probably replace previous two command with a single SQL statement. 

	# remove self from list of columns
	if column in coincidentColumns:
		coincidentColumns.pop(column)

	# get all datasets which contain columns coincident with column
	coincidentDatasets = columnsToDatasets(coincidentColumns.keys(), cnxn)

	# remove all datasets which contain column
	for key in datasetsWithColumn.keys():
		if key in coincidentDatasets:
			coincidentDatasets.pop(key)

	# Get all columns in datasets similar to the datasets with "column", but not containing column
	similarColumns = datasetsToColumns(coincidentDatasets.keys(), cnxn)

	# remove all columns that are coincident with "column"
	for key in coincidentColumns.keys():
		if key in similarColumns:
			similarColumns.pop(key)

	sorted_columns = sorted(similarColumns.items(), key=operator.itemgetter(1))

	nameDict = getColumnNamesFromIds(sorted_columns, cnxn)

	print("---------------------------")
	print("Similar columns:")
	for column in sorted_columns:
		if(column[1] > 1):
			print(str(nameDict[column[0]]) + ": " + str(column[1]))
	print("")
	print("")

# getSimilarDatasets("training-data/CKAN/snnopendata20082009august.csv")
# getSimilarDatasets("training-data/CKAN/table1d-fra-sk.csv")
# getSimilarDatasets("training-data/CKAN/prvtbl1nu.csv")
# getSimilarDatasets("training-data/CKAN/rainfallseptember2015.csv")

# getSimilarColumns("Geographical classification")
getSimilarColumns("550")
# getSimilarDatasets("training-data/CKAN/nndrcurrentreliefaugust.csv")
# getSimilarDatasets("training-data/CKAN/00010012-eng.csv")
# getSimilarDatasets("training-data/CKAN/snnopendata20142015may.csv")
