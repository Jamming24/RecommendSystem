# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
# 分布式tensorflow例子；分布式tensorflow包含三个角色，ps表示服务器，等待各个终端（supervisors）来连接
# worker 在tensorflow的代码注释中称之为supervisors，作为分布式训练的运算终端
# chief supervisors ：在众多运算终端中必须选择一个主要的运算终端，该终端在运算终端中最先启动，功能是
# 合并各个终端运算后的学习参数
# 整个过程都是通过RPC协议通信的

# 定义IP和端口
strps_hosts = "localhost:1681"
strworker_hosts = "localhost:1682,localhost:1683"

# 定义角色名称
strjob_name = "worker"
task_index = 0

# 将字符串转化为数组
ps_hosts = strps_hosts.split(',')
worker_hosts = strworker_hosts.split(',')
cluster_spec = tf.train.ClusterSpec({'ps':ps_hosts, 'worker':worker_hosts})


server = tf.train.Server({'ps':ps_hosts, 'worker':worker_hosts}, job_name=strjob_name, task_index=task_index)



# 创建网络结构
with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index, cluster=cluster_spec)):
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    # 模型参数
    W = tf.Variable(tf.random_normal([1]), name="Weight")
    b = tf.Variable(tf.zeros([1]), name="bias")
    # 获取迭代次数
    global_step = tf.train.get_or_create_global_step()
    # 前向结构
    z = tf.multiply(X, W) + b
    tf.summary.histogram('z', z)# 将预测值以直方图显示
    # 反向优化
    cost = tf.reduce_mean(tf.square(Y - z))
    tf.summary.scalar("loss_function", cost)# 将损失以标量的形式显示
    learning_rate = 0.01
    # 使用梯度下降进行优化
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=1)
    merged_summary_op = tf.summary.merge_all() # 合并所有summary
    # 将之前所设置的变量全部初始化
    init = tf.global_variables_initializer()

# 创建Supervisor，管理Session
# 定义参数
training_epochs = 2200
display_step = 2

sv = tf.train.Supervisor(is_chief=(task_index == 0), # 0表示worker为chief
                        logdir="log/super/", init_op=init, summary_op=None,
                         saver=saver, global_step=global_step, save_model_secs=5)
# save_model_secs=5表示每5秒自动保存一次检查文件点，如果不想保存 则设置为none

# 连接目标角色创建Session
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.rand(*train_X.shape) * 0.3

with sv.managed_session(server.target) as sess:
    print("Sess ok")
    print(global_step.eval(session=sess))

    for epoch in range(global_step.eval(session=sess), training_epochs*len(train_X)):
        print(epoch)
        for (x, y) in zip(train_X, train_Y):
            epoch = sess.run([optimizer, global_step], feed_dict={X:x, Y:y})
            # 生成Summary
            summary_str = sess.run(merged_summary_op,feed_dict={X:x, Y:y})
            # 将Summary写入文件
            # sv.summary_computed(sess, summary_str, global_step=epoch)
            # if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X:train_X, Y: train_Y})
            print("Epoch:", "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))
    print("Finished!")
    sv.saver.save(sess, "log/mnist_with_summaries/"+"sv.cpk", global_step=epoch)
sv.stop()



