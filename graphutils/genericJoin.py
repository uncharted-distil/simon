import pyodbc

def genericJoin(table1_name, table1_value, table2_name, table2_value, join_name, weight, cnxn):
	cursor = cnxn.cursor()

	table1_Id = None
	table2_Id = None

	# check to see if entry 1 already exists
	cursor.execute("SELECT * FROM " + table1_name + " WHERE name = CAST(? as varchar(max))", str(table1_value))
	row = cursor.fetchone()
	if row:
		table1_Id = row[0]
	else:
		# insert if entry 1 does not exist
		cursor.execute("INSERT INTO " + table1_name + " (name) Output Inserted.id VALUES(?)", str(table1_value))
		row = cursor.fetchone()
		table1_Id = row[0]

	# check to see if entry 2 already exists
	cursor.execute("SELECT * FROM " + table2_name + " WHERE name = CAST(? as varchar(max))", str(table2_value))
	row = cursor.fetchone()
	if row:
		table2_Id = row[0]
	else:
		# insert if entry 2 does not exist
		cursor.execute("INSERT INTO " + table2_name + " (name) Output Inserted.id VALUES(?)", str(table2_value))
		row = cursor.fetchone()
		table2_Id = row[0]

	# also create join if that does not exist
	selectJoinQuery = "SELECT * FROM ? WHERE id_1 = ? AND id_2 = ?"
	insertJoinQuery = "INSERT INTO ? VALUES(?,?,?)"

	cursor.execute("SELECT * FROM " + join_name + " WHERE id_1 = ? AND id_2 = ?", table1_Id, table2_Id)
	row = cursor.fetchone()
	if not row:
		cursor.execute("INSERT INTO " + join_name + " VALUES(?,?,?)", table1_Id, table2_Id, weight)

	cursor.commit()
