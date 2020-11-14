# XXXX Aug2: BQTerrace Percentage Qp=32 has an error
import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# input QP 
qp='22';


# poc
poc = '18';


# show bqTerrace all Qps (especially Qp 22 and 27)
# show blowingBubbles 27 for ref2Bits, order 1, qp = 22 order 2


# August2: 
# Show RacehorsesD at QP=22-37
# Show BQTerrace at QP=22, 27

# plan order
order = 1    # 1: linear, 2: quadratic

# input sequence'
# sequenceName='RacehorsesD';
# sequenceName='BasketballPass';
# sequenceName='BlowingBubbles';
sequenceName='BQTerrace';
# sequenceName='BQSquare';

# SHOW BASKETBALLPASS FRAME 17 QP 22 ... DAROORY!

# General path to files
path_to_files = '/Users/hossam.amer/7aS7aS_Works/work/workspace/TESTS/SC_t_star_enc/bin/Build/Products/Release/Gen/Seq-TXT/'
# path_to_files = path_to_files + 'twoRefsModel/frame' + poc + '-dAndR-HM/'
# path_to_files = path_to_files + 'twoRefsModel/frame20-dAndR/'
# path_to_files = path_to_files + 'twoRefsModel/frame17-dAndR/'
# path_to_files = path_to_files + 'twoRefsModel/frame18-dAndR/'
# path_to_files = path_to_files + 'twoRefsModel/frame19-dAndR/'
# path_to_files = path_to_files + 'twoRefsModel/frame20-dAndR/'
path_to_files = path_to_files + 'twoRefsModel/frame17-dAndR-HM-Perc/'


# MSE of reference frame 1
txt_file = path_to_files + sequenceName + '_ref1MSE-' + qp + '.txt'
a = np.loadtxt(txt_file);

# MSE of reference frame 2
txt_file  = path_to_files + sequenceName + '_ref2MSE-' + qp + '.txt'
b = np.loadtxt(txt_file);

# Number of bits for predicted frame (affected by region 1)
#txt_file  = path_to_files + sequenceName + '_ref1Bits-' + qp + '.txt'
txt_file  = path_to_files + sequenceName + '_ref1Bits-Perc-' + qp + '.txt'




# Number of bits for predicted frame (affected by region 2)
# txt_file  = path_to_files + sequenceName + '_ref2Bits-' + qp + '.txt'
# txt_file  = path_to_files + sequenceName + '_ref2Bits-Perc-' + qp + '.txt'


c = np.loadtxt(txt_file);

######################################################## 
# Total rate: 
######################################################## 
# MSE of reference frame 1
path_to_files = '/Users/hossam.amer/7aS7aS_Works/work/workspace/TESTS/SC_t_star_enc/bin/Build/Products/Release/Gen/Seq-TXT/'
path_to_files = path_to_files + 'twoRefsModel/frame' + poc + '-dAndR-HM/'
txt_file= path_to_files+ sequenceName + '_ref1MSE-' + qp + '.txt'
a = np.loadtxt(txt_file);

# MSE of reference frame 2
txt_file= path_to_files+ sequenceName + '_ref2MSE-' + qp + '.txt'
b = np.loadtxt(txt_file);

# Number of bits for predicted frame (affected by region 1)
txt_file= path_to_files+ sequenceName + '_ref1Bits-' + qp + '.txt'
c = np.loadtxt(txt_file);
######################################################## 
######################################################## 



print ('General path file: ', path_to_files)

print ('a shape: ', a.shape)
print ('b shape: ', b.shape)
print ('c shape: ', c.shape)

# print (a)
# print (b)
# print (c)



c = c[:a.shape[0]]
print ('Shorten c: ', c.shape)

# some 3-dim points
mean = np.array([0.0,0.0,0.0])
cov = np.array([[1.0,-0.5,0.8], [-0.5,1.1,0.0], [0.8,0.0,1.0]])
# data = np.random.multivariate_normal(mean, cov, 50)

data = np.c_[a, b, c];
print ('Data Shape: ', data.shape)


# regular grid covering the domain of the data
# X,Y = np.meshgrid(np.arange(-3.0, 3.0, 0.5), np.arange(-3.0, 3.0, 0.5))
X,Y = np.meshgrid(np.arange(-1.0, 30.0, 0.5), np.arange(-1.0, 30.0, 0.5))
XX = X.flatten()
YY = Y.flatten()


if order == 1:
    # best-fit linear plane
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
    
    # evaluate it on grid
    Z = C[0]*X + C[1]*Y + C[2]
    
    # or expressed using matrix/vector product
    #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    print ('Plane Equation is: \n' + str(C[0]) + '*X + ' + str(C[1]) + '*Y + ' + str(C[2]))

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
    
    # evaluate it on a grid
    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)

fig = []
ax = []
n = 3
for i in range(n) :
    fig.append(plt.figure(i))
    ax.append(fig[i].gca())

# plot points and fitted surface
fig = plt.figure(0)
ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2, color='g')
ax.plot_surface(X, Y, Z, alpha=0.2, color='g')
ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
plt.xlabel('Dref1')
plt.ylabel('Dref2')
plt.title(sequenceName + '-' + qp)
ax.set_zlabel('Number of bits for predicted frame')
ax.axis('equal')
ax.axis('tight')
# plt.show()

# Plot X, Z
fig = plt.figure(1)
ax = fig.gca();
# plt.scatter(data[:, 0], data[:, 2], c='r')
ax.scatter(data[:, 0], data[:, 2], c='r')
plt.xlabel('Dref1')
plt.ylabel('Number of bits for predicted frame')


# Plot Y, Z
fig = plt.figure(2)
ax = fig.gca();
ax.scatter(data[:, 1], data[:, 2], c='r')
# plt.scatter(data[:, 1], data[:, 2], c='r')
plt.xlabel('Dref2')
plt.ylabel('Number of bits for predicted frame')

# Show the plot
plt.show()

