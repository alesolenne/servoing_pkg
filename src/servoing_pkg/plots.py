import matplotlib.pyplot as plt
import numpy as np

def plot_results(q_plot, dq_plot, e_plot, i, f):
    """PBVS Visual servoing plot for joints position, velocity and error vector.
    """
    q_plot = q_plot[0:i+1,:]
    dq_plot = dq_plot[0:i+1,:]
    e_plot = e_plot[0:i+1,:]
   
    x = np.zeros((1,i+1))
    for c in range (1,(i+1),1):
        x[:,c] = x[:,c-1]+1/f
    x = x[0]

    plt.figure(1)
    plt.plot(x[0:i], q_plot[0:i,0], color = 'r', label = 'q1')
    plt.plot(x[0:i], q_plot[0:i,1], color = 'g', label = 'q2')
    plt.plot(x[0:i], q_plot[0:i,2], color = 'b', label = 'q3')
    plt.plot(x[0:i], q_plot[0:i,3], color = 'y', label = 'q4')
    plt.plot(x[0:i], q_plot[0:i,4], color = 'c', label = 'q5')
    plt.plot(x[0:i], q_plot[0:i,5], color = 'm', label = 'q6')
    plt.plot(x[0:i], q_plot[0:i,6], color = 'k', label = 'q7')
    plt.title("Andamento posizione giunti")
    plt.grid()
    plt.xlabel("tempo [s]")
    plt.ylabel("posizione giunto [rad]")
    plt.legend(loc = 'upper right')

    plt.figure(2)
    plt.plot(x, dq_plot[:,0], color = 'r', label = 'dq1')
    plt.plot(x, dq_plot[:,1], color = 'g', label = 'dq2')
    plt.plot(x, dq_plot[:,2], color = 'b', label = 'dq3')
    plt.plot(x, dq_plot[:,3], color = 'y', label = 'dq4')
    plt.plot(x, dq_plot[:,4], color = 'c', label = 'dq5')
    plt.plot(x, dq_plot[:,5], color = 'm', label = 'dq6')
    plt.plot(x, dq_plot[:,6], color = 'k', label = 'dq7')
    plt.title("Andamento velocita giunti")
    plt.grid()
    plt.xlabel("tempo [s]")
    plt.ylabel("velocita giunto [rad/s]")
    plt.legend(loc = 'upper right')

    plt.figure(3)
    plt.plot(x, e_plot[:,0], color = 'r', label = 'e1')
    plt.plot(x, e_plot[:,1], color = 'g', label = 'e2')
    plt.plot(x, e_plot[:,2], color = 'b', label = 'e3')
    plt.plot(x, e_plot[:,3], color = 'y', label = 'e4')
    plt.plot(x, e_plot[:,4], color = 'c', label = 'e5')
    plt.plot(x, e_plot[:,5], color = 'm', label = 'e6')
    plt.title("Andamento errore nel tempo")
    plt.grid()
    plt.xlabel("tempo [s]")
    plt.ylabel("errore posizione [m] e orientamento")
    plt.legend(loc = 'upper right')
    plt.show()