#coding=utf-8
# from __future__ import print_function
import tensorflow as tf
# 创建基本操作，是个常量
a = tf.constant(2)
b = tf.constant(3)

#eg1  Launch the default graph.启动默认图形
with tf.Session() as sess:
    print("a=2, b=3")
    print("Addition with constants: %i" % sess.run(a+b))
    print("Multiplication with constants: %i" % sess.run(a*b))


#eg2  tf Graph input(输入占位符)
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operations（定义一些操作）
add = tf.add(a, b)
mul = tf.multiply(a, b)

# Launch the default graph.#又重新启动一个会话
with tf.Session() as sess:
    # Run every operation with variable input用变量输入运行每个操作
    print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))

# Create a Constant op that produces a 1x2 matrix. The op is added as a node to the default graph.
# Create another Constant that produces a 2x1 matrix.
# The returned value, 'product', represents the result of the matrix multiplication.
# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.

#eg3
matrix1 = tf.constant([[3., 3.]])#两个[][],矩阵！！
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
"""
(1)运行matmul我们称之为会话run（）方法，传递'product'这代表了matmul的输出。
(2)这表明我们希望得到matmul op的输出。
(3)该操作所需的所有输入都由会话自动运行。 他们通常是并行运行。
(4)run（product）”的调用因此导致图中三个ops的执行：两个常量和matmul。
(5)op的输出作为一个numpy的`ndarray`对象在'result'中返回。
"""
# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# The call 'run(product)' thus causes the execution of threes ops in the
# graph: the two constants and matmul.
#
# The output of the op is returned in 'result' as a numpy `ndarray` object.
# ==> [[ 12.]]

