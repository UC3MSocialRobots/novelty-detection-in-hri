import matplotlib.pyplot as plt
import user_data_loader as udl
import pandas as pd

# draw a vector
# retrieved from: http://stackoverflow.com/a/11156353/630598
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

# Specifying all the links between joints:
def get_links(data):
    ''' Returns a dict with all the links of the skeleton. 
        Arguments:
            data: a pd.DataFrame with columns with joint names'''
    return { 'head_neck': list(zip(data.T['head'].values, data.T['neck'].values)),
                'neck_lshoulder': zip(data.T['neck'].values, data.T['left_shoulder'].values),
                'lshoulder_lelbow': zip(data.T['left_shoulder'].values, data.T['left_elbow'].values),
                'lelbow_lhand': zip(data.T['left_elbow'].values, data.T['left_hand'].values),
                'neck_rshoulder': zip(data.T['neck'].values, data.T['right_shoulder'].values),
                'rshoulder_relbow': zip(data.T['right_shoulder'].values, data.T['right_elbow'].values),
                'relbow_rhand': zip(data.T['right_elbow'].values, data.T['right_hand'].values),
                'lshoulder_torso': zip(data.T['left_shoulder'].values, data.T['torso'].values),
                'rshoulder_torso': zip(data.T['right_shoulder'].values, data.T['torso'].values),
                'torso_lhip': zip(data.T['torso'].values, data.T['left_hip'].values),
                'lhip_rhip': zip(data.T['left_hip'].values, data.T['right_hip'].values),
                'lhip_lknee': zip(data.T['left_hip'].values, data.T['left_knee'].values),
                'lknee_lfoot': zip(data.T['left_knee'].values, data.T['left_foot'].values),
                'torso_rhip': zip(data.T['torso'].values, data.T['right_hip'].values),
                'rhip_rknee': zip(data.T['right_hip'].values, data.T['right_knee'].values),
                'rknee_rfoot': zip(data.T['right_knee'].values, data.T['right_foot'].values)
            }

def plot_skeleton(axis, datapoints, links=False, **kwargs):
    ''' Plots a skeleton in 3D'''
    axis.scatter(datapoints.x, datapoints.y, datapoints.z, **kwargs)
    if links:
        joint_links = get_links(datapoints)
        for jl in joint_links:                      # adding joint links to the plot:
            arrow = Arrow3D(joint_links[jl][0], joint_links[jl][1], joint_links[jl][2], lw=1, arrowstyle="-", **kwargs)
            axis.add_artist(arrow)

def to_xyz(series, colors=None):
    ''' converts series with index = head_pos_x, head_pos_y, etc... 
        to a dataframe with index=joints and columns = x, y z '''
    def_colors = {'STAND_POINTING_RIGHT':'red', 'STAND_POINTING_FORWARD':'green', 'STAND_POINTING_LEFT':'blue'}
    c = colors if colors else def_colors
    xyz = pd.DataFrame(index=udl.joints, columns=['x','y','z', 'color'])
    x = series[udl.ind_pos_x]
    y = series[udl.ind_pos_y]
    z = series[udl.ind_pos_z]
    for d in (x,y,z): # renaming index so it is the same as xyz
        d.index = udl.joints
    xyz.x, xyz.y, xyz.z = x, y, z
    xyz.color = c[series[-1]]
    return xyz
    
def irow_to_xyz(irow, **kwargs):
    ''' Helper function to pass the pd.iterrows tuple to the to_xyz function '''
    return to_xyz(irow[1], **kwargs)

def df_to_xyz(df):
    ''' converts a a pd.Dataframe with user data to a '3D-plottable' dataframe '''
    return pd.concat(map(irow_to_xyz, df.iterrows()))


def normalize_user(user):
	'''
		returns a normalized user
	'''
	uf = udl.load_user_file('data/exp03-user'+str(user).zfill(2)+'.arff')
    
	multiind_first, multiind_second = udl.make_multiindex(udl.joints, udl.attribs)
	uf.columns = pd.MultiIndex.from_arrays([list(multiind_first), list(multiind_second)], names=['joint', 'attrib'])
	orig_torso, df_normalized = udl.normalize_joints(uf, 'torso')

	uf.update(df_normalized)
	uf.torso = uf.torso - uf.torso
	uf.columns = udl.index
	return uf

def print_users(users, ax):

	'''
		Returns an ax plotted with all the users from a [[number_user, pose]...] list
	'''
	normalized_users = []
	
	for u in users:
		if u[1] =='STAND_POINTING_RIGHT': u[1] = 'red'
		if u[1] =='STAND_POINTING_LEFT': u[1] =  'blue'
		if u[1] =='STAND_POINTING_FORWARD' : u[1] = 'green'
		normalized_users.append([normalize_user(u[0]), u[1]])

	for uf in normalized_users:

		xyz_03u01 = df_to_xyz(uf[0])

		#clouds = xyz_03u01.groupby('color') # Discomment to plot the users clouds as well

		# Plot skeleton joints and links
		means = uf[0].groupby('pose').mean()
		means.insert(len(means.columns), 'pose', means.index )
		# Prepare means to be printed:
		m_groups = [to_xyz(means.ix[i]) for i,ind in enumerate(means.index)]

		for m in m_groups:
	 	    	col = m['color'][0] # Just need the 1st one
	     		if col == uf[1]:
	         		plot_skeleton(ax, m, links=True, color=col)
	         		ske = m 


	plot_skeleton(ax, ske, links=True, color='black') # Plot the last user with black dots to differentiate ir


	'''
	Discomment to plot the user clouds as well

		for c, values in clouds:
		    if c == uf[1]:
		        ax.scatter(values.x, values.y, values.z, color=c, alpha=0.2, marker='o')




	for c, values in clouds:
		if c == uf[1]:
			ax.scatter(values.x, values.y, values.z, color='black', alpha=0.2, marker='x')

	'''	

	ax.view_init(-90,90)

	return ax


def print_user(i, pose):

	if pose =='STAND_POINTING_RIGHT': pose = 'red'
	if pose =='STAND_POINTING_LEFT': pose =  'blue'
	if pose =='STAND_POINTING_FORWARD' : pose ='green'

	uf = normalize_user(i)

	xyz_03u01 = df_to_xyz(uf)

	clouds = xyz_03u01.groupby('color')

	means = uf.groupby('pose').mean()
	means.insert(len(means.columns), 'pose', means.index )
	# Prepare means to be printed:
	m_groups = [to_xyz(means.ix[i]) for i,ind in enumerate(means.index)]

	# Plot skeleton joints and links
	ax = plt.axes(projection='3d')

	for c, values in clouds:
	    if c == pose:
	        ax.scatter(values.x, values.y, values.z, color=c, alpha=0.2, marker='o')
	    
	for m in m_groups:
	    col = m['color'][0] # Just need the 1st one
	    if col == pose:
	        plot_skeleton(ax, m, links=True, color=pose)

	ax.view_init(-90,90)
	#plt.savefig('/Users/almudenasanz/Downloads/skeleton.pdf', format='pdf')