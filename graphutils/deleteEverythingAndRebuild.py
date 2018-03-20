import pyodbc
import getConnection as gc

# cnxn = gc.getConnection()

# cursor = cnxn.cursor()

# cursor.execute("DROP TABLE columns_datasets")
# cursor.execute("DROP TABLE columns_columnnames")
# cursor.execute("DROP TABLE columns_columnlabels")

# cursor.commit()

# cursor.execute("DROP TABLE columns")
# cursor.execute("DROP TABLE datasets")
# cursor.execute("DROP TABLE columnnames")
# cursor.execute("DROP TABLE columnlabels")

# cursor.commit()

# cursor.execute("CREATE TABLE columns (id BIGINT NOT NULL IDENTITY(1,1) PRIMARY KEY, name varchar(max));")
# cursor.execute("CREATE TABLE datasets (id BIGINT NOT NULL IDENTITY(1,1) PRIMARY KEY, name varchar(max));")

# cursor.commit()

# cursor.execute("CREATE TABLE columns_datasets (id_1 BIGINT FOREIGN KEY REFERENCES columns(id), id_2 BIGINT FOREIGN KEY REFERENCES datasets(id), weight smallint);")

# cursor.commit()

# cursor.execute("CREATE TABLE columnnames (id BIGINT NOT NULL IDENTITY(1,1) PRIMARY KEY, name varchar(max));")

# cursor.commit()

# cursor.execute("CREATE TABLE columns_columnnames (id_1 BIGINT FOREIGN KEY REFERENCES columns(id), id_2 BIGINT FOREIGN KEY REFERENCES columnnames(id), weight smallint);")

# cursor.commit()

# cursor.execute("CREATE TABLE columnlabels (id BIGINT NOT NULL IDENTITY(1,1) PRIMARY KEY, name varchar(max));")

# cursor.commit()

# cursor.execute("CREATE TABLE columns_columnlabels (id_1 BIGINT FOREIGN KEY REFERENCES columns(id), id_2 BIGINT FOREIGN KEY REFERENCES columnlabels(id), weight smallint);")

# cursor.commit()

# cnxn.close()