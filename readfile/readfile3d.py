import numpy as np

def get_raw_data(input_name, writebox, spaces=False):

	f = open(input_name+'.txt', 'r')
	row = f.readline()

	points = []
	for row in f:
		points.append(row.strip('\n').split(','))
	f.close()

	points = np.array(points, dtype='float')

	return points 

def get_sequence(sequence, writebox, spaces=False):
    """Returns a sequence 

    sequence list: the sequence of integers
    writebox list: [min x, max x, min y, max y]
    """

    nans = np.array([np.nan, np.nan, np.nan])
    nums= nans.copy()

    for ii, nn in enumerate(sequence):
        if isinstance(nn, int):
            nn = str(nn)
        num = get_raw_data(nn, writebox)
        nums = np.vstack([nums, num, nans])

    return nums 

### Testing code ###
if __name__ == '__main__':

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')
	#files=['h','e','l','l','o','w','o','r','l','d']
    files=['3d']
    nums = get_sequence(files, [-1,1,0,1], spaces=False)
    ax.plot(nums[:,2], nums[:,1], nums[:,0], label='demo')
    ax.legend()
    plt.show()
    # print(type(nums))
    # plt.plot(nums[:,2], nums[:,1], nums[:,0])
    # plt.axis('off')
    # plt.show()