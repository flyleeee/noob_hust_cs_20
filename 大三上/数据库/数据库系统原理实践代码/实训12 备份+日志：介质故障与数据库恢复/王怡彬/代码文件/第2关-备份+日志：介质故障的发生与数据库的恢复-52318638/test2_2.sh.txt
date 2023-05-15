# 这是shell脚本，将在linux命令行上执行
# 命令行上可省略密码的指定
# 请写出利用逻辑备份和日志恢复数据库的命令：

mysql -h127.0.0.1 -uroot < train_bak.sql;
mysqlbinlog --no-defaults log/binlog.000018 | mysql -uroot;

