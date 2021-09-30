import tensorflow as tf
import numpy as np
import time, datetime
from tensorflow.python.training import moving_averages
#This is Black-Scholes-Barenblatt Equation problem
import time, datetime
import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages
from tensorflow.keras.models import *
start_time=time.time()
# tf.reset_default_graph()
name='BSB'
# d=4
d=100
# batch_size=2
batch_size=64
T=1.0
N=20
h=T/N
sqrth=np.sqrt(h)
# print(h)
n_maxstep=500
n_displaystep=100
n_neuronForA=[d,d,d,d]
n_neuronForGamma=[d,d,d,d**2]
Xinit=np.array([1.0,0.5]*50)
# print("This is shape of Xinit")
# print(Xinit)
# print(Xinit.shape)
mu=0
sigma=0.4
sigma_min=0.1
sigma_max=0.4
r=0.05
_extra_train_ops=[]
def sigma_value(W):
    return sigma_max*tf.cast(tf.greater_equal(W,tf.cast(0,tf.float64)),tf.float64) +\
    sigma_min*tf.cast(tf.greater(tf.cast(0,tf.float64),W),tf.float64)

def f_tf(t,X,Y,Z,Gamma):
    return -0.5*tf.expand_dims(
        tf.linalg.trace(
          tf.square(tf.expand_dims(X,-1))*(tf.square(sigma_value(Gamma))-sigma**2)*Gamma),-1) +\
          r*(Y-tf.math.reduce_sum(X*Z,1,keepdims=True))

def g_tf(X):
    return tf.math.reduce_sum(tf.square(X),1,keepdims=True)

def sigma_function(X):
    return sigma*tf.linalg.diag(X)

def mu_function(X):
    return mu*X

def _one_time_net(x,name,isgamma=False):
    with tf.compat.v1.variable_scope(name):
        x_norm=_batch_norm(x,name='layer0_normalization')
        layer1=_one_layer(x_norm,(1-isgamma)*n_neuronForA[1]+isgamma*n_neuronForGamma[1],name='layer1')
        layer2=_one_layer(x_norm,(1-isgamma)*n_neuronForA[2]+isgamma*n_neuronForGamma[2],name='layer2')
        z=_one_layer(layer2, (1-isgamma)*n_neuronForA[3]+isgamma*n_neuronForGamma[3],activation_fn=None,name='final')
    return z
  # Defining layers
def _one_layer(input_,output_size,activation_fn=tf.nn.relu,stddev=5.0, name='linear'):
    with tf.compat.v1.variable_scope(name):
        shape=input_.get_shape().as_list()
        w=tf.compat.v1.get_variable('Matrix',[shape[1],output_size],tf.float64,tf.random_normal_initializer(stddev=stddev/np.sqrt(shape[1]+output_size)))
        hidden=tf.matmul(input_,w)
        hidden_bn=_batch_norm(hidden,name='normalization')
    if activation_fn:
        return activation_fn(hidden_bn)
    else:
        return hidden_bn

def _batch_norm(x,name):
    with tf.compat.v1.variable_scope(name):
        params_shape=[x.get_shape()[-1]]
        beta=tf.compat.v1.get_variable('beta',params_shape,tf.float64,initializer=tf.random_normal_initializer(0.0,stddev=0.1))
        gamma=tf.compat.v1.get_variable('beta',params_shape,tf.float64,initializer=tf.random_uniform_initializer(0.1,0.5))
        moving_mean=tf.compat.v1.get_variable('moving_mean',params_shape,tf.float64,initializer=tf.constant_initializer(0.0),trainable=False)
        moving_variance=tf.compat.v1.get_variable('moving_variance',params_shape,tf.float64,initializer=tf.constant_initializer(1.0),trainable=False)
        mean,variance=tf.nn.moments(x,[0],name='moments')
        _extra_train_ops.append(moving_averages.assign_moving_average(moving_mean,mean,0.99))
        _extra_train_ops.append(moving_averages.assign_moving_average(moving_variance,variance,0.99))
        y=tf.nn.batch_normalization(x,mean,variance,beta,gamma,1e-6)
        return y

# one=_one_layer(X,n_neuronForGamma[1])
# print(one)
# print(X)
# two=_one_time_net(X,'two',isgamma=True)
# print(two)
with tf.compat.v1.Session() as sess:
    dW=tf.random.normal(shape=[batch_size,d],stddev=1,dtype=tf.float64)
    X=tf.Variable(np.ones([batch_size,d]))*Xinit
    Y0=tf.Variable(tf.random.uniform([1], minval=0, maxval=1,dtype=tf.float64),name='Y0')
    Z0=tf.Variable(tf.random.uniform([1,d],minval=-0.1,maxval=0.1,dtype=tf.float64),name='Z0')
    Gamma0=tf.Variable(tf.random.uniform([d,d],minval=-1,maxval=1,dtype=tf.float64),name='Gamma0')
    A0=tf.Variable(tf.random.uniform([1,d],minval=-0.1,maxval=0.1,dtype=tf.float64),name='A0')
    allones=tf.ones(shape=[batch_size,1],dtype=tf.float64,name='MatrixOfONes')
    Y=allones*Y0
    Z=tf.matmul(allones,Z0)
    A=tf.matmul(allones,A0)
    Gamma=tf.multiply(tf.ones(shape=[batch_size,d,d],dtype=tf.float64),Gamma0)

    with tf.compat.v1.variable_scope('foward', reuse=tf.compat.v1.AUTO_REUSE):
        for t in range(0,N-1):
            dX=mu*X*h+sqrth*sigma*X*dW
            Y=Y+f_tf(t*h,X,Y,Z,Gamma)*h + tf.math.reduce_sum(Z*dX,1,keepdims=True)
            X=X+dX
            Z=Z+A*h+tf.squeeze(tf.matmul(Gamma,tf.expand_dims(dX,-1),transpose_b=False))
            A=_one_time_net(X,str(t+1)+"A")/d
            Gamma=_one_time_net(X,str(t+1)+"Gamma",isgamma=True)/d**2
            Gamma=tf.reshape(Gamma,[batch_size,d,d])
            dW=tf.random.normal(shape=[batch_size,d],stddev=1,dtype=tf.float64)
    #Y update outside of the loop - terminal time step
        dX=mu*X*h+sqrth*sigma*X*dW
        Y=Y+f_tf((N-1)*h,X,Y,Z,Gamma)*h+tf.math.reduce_sum(Z*dX,1,keepdims=True)
        X=X+dX
        loss=tf.reduce_mean(tf.square(Y-g_tf(X)))

  #training operations
    global_step=tf.compat.v1.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False, dtype=tf.int32)
    learning_rate=tf.compat.v1.train.exponential_decay(1.0,global_step,decay_steps=200,decay_rate=0.5,staircase=True)
    trainable_variables=tf.compat.v1.trainable_variables()
    with tf.GradientTape() as tape:
        loss=loss
    grads=tape.gradient(loss,trainable_variables)
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    apply_op=optimizer.apply_gradients(zip(grads,trainable_variables),global_step=global_step,name='train_step')
    train_ops=[apply_op]+ _extra_train_ops
    train_op=tf.group(*train_ops)

    with tf.control_dependencies([train_op]):
        train_op_2=tf.identity(loss,name='train_op2')

  # to save history
    learning_rates=[]
    y0_values=[]
    losses=[]
    running_time=[]
    steps=[]
    sess.run(tf.global_varaibles_initializer())
    try:
        for _ in range(n_maxstep+1):
            step, y0_value=sess.run([global_step,Y0])
            currentLoss,currentLearningRate=sess.run([train_op_2,learning_rate])
            steps.append(step)
            losses.append((currentLoss))
            y0_values.append(y0_value)
            learning_rates.append(currentLearningRate)
            running_time.append(time.time()-start_time)

            if step% n_displaystep==0:
                print("step: ", step,
                    "loss: ", currentLoss,
                    " Y0: ", y0_value,
                    " learning rate: ", currentLearningRate)
            end_time=time.time()
            print("running time: ", end_time-start_time)

    except KeyboardInterrupt:
        print("Mannually disengaged")
#writing results to a csv file
output=np.zeors((len(y0_values),5))
output[:,0]=steps
output[:,1]=losses
output[:,2]=y0_values
output[:,3]=learning_rates
output[:,4]=running_time
print("Result saved at "+str(name) + "_d"+str(d)+"_"+datetime.datetime.now().strftime('%Y-%m-%d-%H:%M;%S')+".csv")
np.savetxt("./"+str(name) + "_d"+str(d)+"_"+datetime.datetime.now().strftime('%Y-%m-%d-%H:%M;%S')+".csv", output, delimiter=", ", header="step, loss function, Y0, learning rate, running time")
