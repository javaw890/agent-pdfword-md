from pymongo import MongoClient

myclient = MongoClient("mongodb://192.168.10.100:27017/")

dblist = myclient.list_database_names()
# dblist = myclient.database_names()
if "runoobdb" in dblist:
    print("数据库已存在！")
print(dblist)