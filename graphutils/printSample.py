import pyodbc

def printSample(cursor):
	cursor.execute("SELECT top(10) * FROM columns")
	columns = cursor.fetchall()
	print("COLUMNS SAMPLE:")
	print("---------")
	for column in columns:
		print(str(column[0]) + ", " + str(column[1]))
	print("")
	
	cursor.execute("SELECT top(10) * FROM datasets")
	datasets = cursor.fetchall()
	print("DATASETS:")
	print("---------")
	for dataset in datasets:
		print(str(dataset[0]) + ", " + str(dataset[1]))
	print("")

	cursor.execute("SELECT top(10) * FROM columns_datasets")
	joins = cursor.fetchall()
	print("JOINS: ")
	print("---------")
	for join in joins:
		print(str(join[0]) + ", " + str(join[1]))
	print("")