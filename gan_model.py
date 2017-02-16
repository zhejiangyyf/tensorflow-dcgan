# -*- coding: utf8 -*-

import tensorflow as tf
from numpy import random
class GanModel:
    def __init__(self,batch_num=128):
        self.batch_num=batch_num
    # define weight variable
    def weight_variable(self,shape,name="weights"):
        initial=tf.truncated_normal(shape,stddev=0.01)
        weights=tf.get_variable(name,initializer=initial)
        return weights
    # define bias variable
    def bias_variable(self,shape,name="biases"):
        initial=tf.constant(0.0,shape=shape)
        bias=tf.get_variable(name,initializer=initial)
        return bias

    ### define layers ###
    # linear,fc
    def linear(self,x,input_num,output_num):
        w=self.weight_variable((input_num,output_num),name="weights")
        b=self.bias_variable([output_num],name="biases")
        return tf.matmul(x,w)+b
    # convolution2d
    def conv2d(self,x,input_num,output_num,kernel_size=3,stride=1,padding="SAME"):
        # stride is 2 as paple say
        # for instead pooling layer
        # this make future map to half size everytime
        W=self.weight_variable([kernel_size,kernel_size,input_num,output_num],name="weights")
        conv=tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding=padding)
        # add bias
        b=self.bias_variable([output_num],name="biases")
        conv=conv+b
        return conv
    # convolution2d upsample
    def conv2d_transpose(self,value,output_shape,kernel_size,stride):
        # output_shape:[batch_size, height, width, channel]
        # stride is 2 as paple say
        # this make future map to 2 times size everytime
        input_num=int(value.get_shape()[-1])
        output_num=output_shape[-1]
        w=self.weight_variable([kernel_size,kernel_size,output_num,input_num],name="weights")
        deconv=tf.nn.conv2d_transpose(value=value,filter=w,output_shape=output_shape,strides=[1,stride,stride,1])
        # add bias
        b=self.bias_variable([output_num],name="biases")
        deconv=deconv+b
        return deconv
    # batch normalization
    def bn(self,x,output_num):
        mean, variance = tf.nn.moments(x, axes=[0,1,2])
        beta = tf.get_variable(name="beta",initializer=tf.zeros([output_num]))# init as 0
        gamma=tf.get_variable(name="gamma",initializer=tf.ones([output_num]))# init as 1
        batch_norm=tf.nn.batch_normalization(x, mean, variance, beta, gamma,variance_epsilon=0.00001)# variance_epsilon is just for avoid division by zero
        return batch_norm
    def relu(self,x):
        return tf.nn.relu(x)
    # leaky relu
    def lrelu(self,x, leak=0.2):
        return tf.maximum(x, leak*x)
    def sigmoid(self,x):
        return tf.nn.sigmoid(x)
    def tanh(self,x):
        return tf.nn.tanh(x)

    ####### generator ########
    # input z:[batch_num,100]
    def generator(self,z):
        with tf.variable_scope("gen"):
            with tf.variable_scope("defc1"):
                #[b,input_num]=>[b,output_num]
                input_num=100
                output_num=4*4*1024
                defc1=self.linear(z,input_num=input_num,output_num=output_num)
                defc1=tf.reshape(defc1,shape=(self.batch_num,4,4,1024))# future map:[2,4,4,1024]
                defc1=self.bn(defc1,1024)
                defc1=self.relu(defc1)
            with tf.variable_scope("deconv2"):
                deconv1=self.conv2d_transpose(defc1,output_shape=[self.batch_num,8,8,512],kernel_size=5,stride=2)# future map: [2,8,8,512]
                deconv1=self.bn(deconv1,512)
                deconv1=self.relu(deconv1)
            with tf.variable_scope("deconv3"):
                deconv2=self.conv2d_transpose(deconv1,output_shape=[self.batch_num,16,16,256],kernel_size=5,stride=2)# future map: [2,16,16,256]
                deconv2=self.bn(deconv2,256)
                deconv2=self.relu(deconv2)
            with tf.variable_scope("deconv4"):
                deconv3=self.conv2d_transpose(deconv2,output_shape=[self.batch_num,32,32,128],kernel_size=5,stride=2)# future map: [2,32,32,128]
                deconv3=self.bn(deconv3,128)
                deconv3=self.relu(deconv3)
            with tf.variable_scope("deconv5"):
                deconv4=self.conv2d_transpose(deconv3,output_shape=[self.batch_num,64,64,3],kernel_size=5,stride=2)# future map: [2,64,64,3]
                # no batch normalization here as paple say
                deconv4_tanh=self.tanh(deconv4)
        # output scale:[-1,1]
        return deconv4_tanh

    ####### discriminator ########
    # input x:[batch_num,64,64,3]
    def discriminator(self,x):
        with tf.variable_scope("disc"):
            with tf.variable_scope("conv1"):
                conv1=self.conv2d(x,input_num=3,output_num=128,kernel_size=5,stride=2)# future map: [2,32,32,128]
                # no batch normalization here as paple say
                conv1=self.lrelu(conv1)
            with tf.variable_scope("conv2"):
                conv2=self.conv2d(conv1,input_num=128,output_num=256,kernel_size=5,stride=2)# future map: [2,16,16,256]
                conv2=self.bn(conv2,256)
                conv2=self.lrelu(conv2)
            with tf.variable_scope("conv3"):
                conv3=self.conv2d(conv2,input_num=256,output_num=512,kernel_size=5,stride=2)# future map: [2,8,8,512]
                conv3=self.bn(conv3,512)
                conv3=self.lrelu(conv3)
            with tf.variable_scope("conv4"):
                conv4=self.conv2d(conv3,input_num=512,output_num=1024,kernel_size=5,stride=2)# future map: [2,4,4,1024]
                conv4=self.bn(conv4,1024)
                conv4=self.lrelu(conv4)
            with tf.variable_scope("conv5"):
                b,h,w,c=conv4.get_shape()
                input_num=int(h)*int(w)*int(c)
                # flat
                conv5_flat=tf.reshape(conv4,[-1,input_num])
                fc5=self.linear(conv5_flat,input_num=input_num,output_num=1)
                fc5_sigmoid=self.sigmoid(fc5)
        # output scale:[0,1]
        return fc5_sigmoid

def modelrun():
    batch_num=128
    # define model
    model=GanModel(batch_num=batch_num)

    # define placeholder variable
    z=tf.placeholder(tf.float32,shape=(None,100))
    x=tf.placeholder(tf.float32,shape=(None,64,64,3))# images batch,only for 64*64*3 size

    # build the model
    fc5_sigmoid=model.discriminator(x)
    deconv4_tanh=model.generator(z)

    # run model in session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # feed z random data
        z_train=random.rand(batch_num, 100)
        result=sess.run([deconv4_tanh],feed_dict={z:z_train})
        print result[0].shape # (128, 64, 64, 3),generate 128(batch_num) images with size 60*60*3
        # feed image data
        x_train=random.rand(batch_num, 64,64,3)
        result=sess.run([fc5_sigmoid],feed_dict={x:x_train})
        print result[0].shape # (128, 1),judge 128(batch_num) images for ture or false
if __name__=="__main__":
    modelrun()
