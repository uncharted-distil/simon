import azure_utils.client as client
import graphutils.printSample as printSample
import graphutils.getConnection as gc
import graphutils.insertColumnDatasetJoin as insert

import pandas
import sys
import random
import pyodbc

def graphDoesntContainFile(filename,cnxn):
	cursor = cnxn.cursor()

	cursor.execute("SELECT top(1) * FROM datasets where name=?",filename)

	name = cursor.fetchone()

	return name == None

store_name = 'nktraining'
adl = client.get_adl_client(store_name)

files = adl.ls('training-data/CKAN')

random.shuffle(files)

cnxn = gc.getConnection()

i = 0
for file in files:
	if(i > 1000):
		break

	if graphDoesntContainFile(file, cnxn):
		try:
			with adl.open(file, blocksize=2**20) as f:
				if(file.startswith('training-data/CKAN/BroadcastLogs') or file.startswith('training-data/CKAN/barrownndremptyexemption')):
					continue
				if(file.endswith('csv')):
					print("Loading (" + str(i) + "): " + file + " into metadata store")
					frame = pandas.read_csv(f,nrows=3,sep=None)
				# else:
				# 	frame = pandas.read_excel(f)
					for colName in frame.columns:
						if not str(colName).startswith('Unnamed'):
							insert.insertColumnDatasetJoin(colName, file, cnxn)
				i = i + 1
				cnxn.commit()
		except UnicodeEncodeError:
			print("Failed to parse filename")
		except ValueError:
			print("Encountered poorly formatted file: " + str(file))
		except TypeError:
			print("Encountered bad delimiter in: " + str(file))
		except:
			print("It broke and I don't know why, possibly something about newlines " + str(file))
			print(sys.exc_info())
	else:
		print("Skipping " + file + " because it is already in db")

printSample.printSample(cnxn.cursor())

cnxn.close()