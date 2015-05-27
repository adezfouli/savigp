import numpy as np
import numpy.testing
import scipy.io
import learn_predict

def ESS(f, logli, lower_chol_K_compact, read_randoms):
    nu =  lower_chol_K_compact.dot(read_randoms(lower_chol_K_compact.n * lower_chol_K_compact.n_labels + lower_chol_K_compact.n_labels ** 2, 'n')) # sample from K
    u = read_randoms(1, 'u')
#    u = np.random.rand(1)
#    print("initial logli evaluation")
    log_y = numpy.log(u) + logli(f)
    read_randoms(1, should=log_y)
    read_randoms(f.shape[0], should=f)
    v = read_randoms(1, 'u')
    theta = v*2*numpy.pi
#    theta = np.random.rand(1)*2*numpy.pi
    theta_min = theta - 2*numpy.pi
    theta_max = theta
    while True:
        fp = f*np.cos(theta)+nu*np.sin(theta)
        #print("f : %s" % f.dtype)
        #print("nu : %s" % nu.dtype)

#        print("log li evaluation in loop")
        cur_log_like = logli(fp)
        read_randoms(1, should=cur_log_like )
        read_randoms(fp.shape[0], should=fp)
        if (cur_log_like > log_y):
            break
        if (theta < 0):
            theta_min = theta
        else:
            theta_max = theta
        v = read_randoms(1, 'u')
        theta = v*(theta_max - theta_min) + theta_min
#        theta = np.random.rand(1)*(theta_max - theta_min) + theta_min
    return (fp, cur_log_like)
