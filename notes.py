import tensorflow as tf

session = tf.Session()

def run(g, d):
    print(session.run(g))

def run_with_dict(g, d):
    print(session.run(g, feed_dict=d))

def constants():
    a = tf.constant(2) # cf to a = 2 in python, note type error if 2.1
    # a = tf.constant(2.1, dtype=tf.float32)  # can fix type errors with explicit annotation
    b = tf.constant(3)
    # b = tf.constant(3, dtype=tf.float32)
    c = tf.add(a, b)
    print(c)
    # >> Tensor("Add:0", shape=(), dtype=int32)
    print(session.run(c))

# Exercise - Math Operations

def ex1():
    a = tf.constant(3, dtype=tf.float32)
    b = tf.constant(4.1, dtype=tf.float32)
    c = tf.constant(5, dtype=tf.float32)
    m = tf.multiply(a, b)
    r = tf.add(m, c)
    print(session.run(r))

# Exercise - Matrices
def ex2():

    a = tf.constant([
        [1,2],
        [3,4]
    ], dtype=tf.float32)

    b = tf.constant([
        [2,2],
        [2,2]
    ], dtype=tf.float32)

    run(tf.add(a, b))
    run(tf.multiply(a,b))       # elem-wise multiplication
    run(tf.matmul(a,b))         # matrix multiplication
    run(tf.reduce_sum(a,0))     # reduce rows
    run(tf.reduce_sum(a,1))     # reduce cols
    run(tf.argmax(a))           # the index position of the max value

    zeros = tf.zeros([2, 2])
    rnd = tf.random_normal([2,2])
    trnd = tf.truncated_normal([2, 2], stddev=0.1)

def ex2():
    x = tf.constant([
      [1.1, 1.2]
    ])

    w = tf.constant([
        [1,2],
        [3,4]
    ], dtype=tf.float32)

    b = tf.constant([
        [2.3, 2.5]
    ])

    t = tf.matmul(x, w)
    y = tf.add(t, b)
    run(t)
    run(y)

def placeholders():
    # i.e inputs
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    c = tf.add(a,b)
    run_with_dict(c, {a:2, b:3})

def ex2_placeholders():

    x = tf.placeholder(tf.float32, shape=[1,2])

    w = tf.constant([
        [1,2],
        [3,4]
    ], dtype=tf.float32)

    b = tf.constant([
        [2.3, 2.5]
    ])

    t = tf.matmul(x, w)
    y = tf.add(t, b)

    run_with_dict(y, {x: [[1.1, 1.2]]})

def variables():
    # i.e anything you want to learn
    w = tf.Variable([0.], tf.float32)
    b = tf.Variable([0.], tf.float32)
    init = tf.global_variables_initializer
    run(init)

