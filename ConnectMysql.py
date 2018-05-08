# coding=utf-8
import pymysql


# 打开数据库连接（ip/数据库用户名/登录密码/数据库名）
def connetDatabase(Host, userName, passwords, DBname):
    db = pymysql.connect(host=Host, port=3306, user=userName, password=passwords, db=DBname, charset='utf8')
    return db

def test():
    host = "60.205.213.252"
    user = "whshop"
    password = "whshop123"
    db = "test_whshop"
    connetDatabase(host, user, password, db)
    db = connetDatabase(host, user, password, db)
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor(cursor=pymysql.cursors.DictCursor)
    # 使用 execute()  方法执行 SQL 查询
    sql = "select id from wh_pre_portal_article_title where title like \"%哈尔滨%\" "
    cursor.execute(sql)
    # 获取剩余结果所有数据
    row_3 = cursor.fetchall()

    for r in row_3:
        print(r)
    # 关闭游标
    cursor.close()
    # 关闭数据库连接
    db.close()

test()