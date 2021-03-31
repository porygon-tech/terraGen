import numpy as np
import matplotlib.pyplot as plt


def showdata(mat, color=plt.cm.gist_earth):
	mat = np.copy(mat)
	mat[0,0] = 0
	mat[0,1] = 1
	plt.imshow(mat.astype('float32'), interpolation='none', cmap=color)
	plt.show()

def sharpen_slow(x, thres=0.5, spe=1, epochs = 10):
	b=thres/(1+thres)
	for i in range(epochs):
		x += spe * ((b-1) * x**3 + x**2 - b * x)
	return x

def sharpen_fast(x, thres=0.5, spe=1, epochs = 10):
	for i in range(epochs):
		x += spe * x * (thres-x) * (x-1)
	return x

def smooth(land, epochs = 100):
	height, width = land.shape
	for i in range(epochs):
		D = np.delete((np.insert(land, height, 0.5, axis=0)), 0, axis=0)
		U = np.delete((np.insert(land, 0, 0.5, axis=0)), height, axis=0)
		R = np.delete((np.insert(land, width, 0.5, axis=1)), 0, axis=1) 
		L = np.delete((np.insert(land, 0, 0.5, axis=1)), width, axis=1)
		y_smooth = (D + land + U)/3
		x_smooth = (R + land + L)/3
		land = (y_smooth + x_smooth)/2
	return land

def landgen(width,height,smoothing=True,sharpening=None):
	land = np.random.rand(width,height)
	if smoothing:
		land = smooth(land)
	if sharpening == 'slow':
		land = sharpen_slow(land)
	elif sharpening == 'fast':
		land = sharpen_fast(land)
	return(land)

#=================================================================
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D

def terrainplot(land, color=plt.cm.gist_earth, definition=1):
	#land = land[1:-1,1:-1]
	land = np.copy(land)
	land[0,0] = 0
	land[0,1] = 1
	#ls = LightSource(270, 45)
	#rgb = ls.shade(land, cmap=plt.cm.gist_earth, vert_exag=0.1, blend_mode='soft')
	height, width = land.shape
	xi = np.linspace(0, width, width)
	yi = np.linspace(0, height, height)
	x, y = np.meshgrid(xi, yi)
	fig = plt.figure()
	ax = Axes3D(fig)
	#surf = ax.plot_surface(x, y, geo, rstride=1, cstride=1, facecolors=rgb, linewidth=0, antialiased=False, shade=False)
	surf = ax.plot_surface(x, y, land, rcount=height/definition, ccount=width/definition, cmap=color, linewidth=0, antialiased=False)
	plt.show()



#=================================================================
#mountains and lakes
geo = landgen(64,64,sharpening='slow')
#showdata(geo)
#terrainplot(geo)

'''
noise = landgen(128,128,smoothing=False)*0.01
terrainplot(geo+noise)
'''

#=================================================================

'''
geo2 = landgen(256,256)
height, width = geo2.shape
param_map_1 = landgen(height, width)
sharpen_fast(param_map_1,thres=0.4,epochs=3)
smooth(param_map_1,epochs=5)


#CUIDADO PORQUE SHARPEN NO ESTÁ HECHA PARA USARSE ASÍ
geo2 = np.array(list(map(sharpen_slow, geo2, param_map_1,np.repeat(2))))

terrainplot(geo2)
'''


#=================================================================
#add rivers
'''
def rain_erode(land, epochs=100):
	waterlayer = np.ones(land.shape)
'''
#at each step, water finds steepest direction in the 3x3 adjacent grid
#simulation ends when there is no steepest descent left. Then, next epoch starts
#depending on the amount of water (and its velocity), a determined amount of ground is removed:
#rápida: excava hacia abajo
#abundante: excava hacia los lados
#añadir caudal de lluvia


# Crear la función de superficie z(x,y) usando interpolación en la matriz. 
# usar esta función en un modelo determinista de gradiente para simular el flujo del agua.
# ¿Sería posible incluir la erosión para modificar el gradiente a medida que pasa el tiempo y fluye el agua?
# sería algo como que los vectores se acortan cuando son cortos y se alargan cuando son largos. 
# Intuyo que el umbral que separa largos y cortos se va alterando con el tiempo.
# o más fino: puede que la modificación de los vectores con la erosión tenga algo que ver con la divergencia en esa región.
# hablar con Jose


from matplotlib.colors import ListedColormap
# Choose colormap
cmap = plt.cm.Blues
# Get the colormap colors
waterAlpha = cmap(np.arange(cmap.N))
# Set alpha
waterAlpha[:,-1] = np.linspace(0, 1, cmap.N)
# Create new colormap
waterAlpha = ListedColormap(waterAlpha)

def plotmap(land, waters):
	plt.figure()
	plt.imshow(land, 'gist_earth', interpolation='none')
	plt.imshow(waters, cmap, interpolation='none')
	plt.show()

def DURL(mat):
	height, width = mat.shape
	D = np.delete((np.insert(mat, height, 0.5, axis=0)), 0, axis=0)
	U = np.delete((np.insert(mat, 0, 0.5, axis=0)), height, axis=0)
	R = np.delete((np.insert(mat, width, 0.5, axis=1)), 0, axis=1)
	L = np.delete((np.insert(mat, 0, 0.5, axis=1)), width, axis=1)
	return (D,U,R,L)

#interpolation:
def ip_di(x):
	return int(round(-2/3*x**3+7/2*x**2-29/6*x+1))

def ip_dj(x):
	return int(round(-2/3*x**3+5/2*x**2-11/6*x))

def flood(land,waters,epochs=1,tipoA=True):
	if tipoA:
		dD,dU,dR,dL = DURL(land) - land
	waters = np.copy(waters)
	height, width = land.shape
	for t in range(epochs):
		if not tipoA:
			dD,dU,dR,dL = DURL(land+waters) - (land+waters)
		w_new = (np.reshape(np.zeros(height*width), (height,width)))
		for i in range(height):
			for j in range(width):
				slopes = np.array((dD[i,j],dU[i,j],dR[i,j],dL[i,j]))	
				if waters[i,j] != 0:
					if (slopes < 0).sum() != 0:	
						split = waters[i,j]/(slopes < 0).sum()
						directions = np.where(slopes < 0)[0]
						for d in directions:
							w_new[i+ip_di(d),j+ip_dj(d)] += split
						waters[i,j] = 0	
					else:
						w_new[i,j] = waters[i,j]
		waters = np.copy(w_new)
	return(waters)

#---------------------------------

#
#waterlayer = np.reshape(np.ones(height*width), (height,width))
#dD,dU,dR,dL = DURL(geo+waterlayer) - (geo+waterlayer)

#waterlayer = np.reshape(np.ones(height*width), (height,width))

waterlayer_initdeep = np.mean(np.abs(DURL(geo)-geo)) #initial value for the waterlayer


height, width = geo.shape
waterlayer = (np.reshape(np.zeros(height*width), (height,width)))
waterlayer[16:height-16,16:width-16] = waterlayer_initdeep * 10

plotmap(geo,flood(geo,waterlayer,4))
plotmap(geo,flood(geo,waterlayer,4,False))

showdata(flood(geo,waterlayer,4,False),'Blues')
showdata(flood(geo,waterlayer,10,False))
#waterlayer = landgen(height,width)
showdata(waterlayer, 'Blues')



#showdata(waterlayer, 'Blues')


#showdata(waterlayer, 'Blues')




#showdata(waterlayer, 'Blues')

'''
from matplotlib.colors import ListedColormap
cmap = pl.cm.Blues
my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
my_cmap = ListedColormap(my_cmap)
'''


waterlayer += np.reshape(np.ones(height*width), (height,width)) + w_new



showdata(waterlayer, 'Blues')

geo = smooth(waterlayer, epochs=7)

terrainplot(geo)

