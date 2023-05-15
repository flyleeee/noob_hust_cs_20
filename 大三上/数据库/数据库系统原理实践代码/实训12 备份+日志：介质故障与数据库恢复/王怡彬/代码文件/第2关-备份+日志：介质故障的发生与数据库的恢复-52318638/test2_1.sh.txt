# 这是shell脚本，将在linux命令行上执行
# 命令行上可省略密码的指定
# 请写出对数据库train作逻辑备份并新开日志文件的命令，备份文件你可以自己命名(如train_bak.sql)：

mysqldump -h127.0.0.1 -uroot --flush-logs --databases train > train_bak.sql;


