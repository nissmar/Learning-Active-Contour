"""source: https://github.com/xuuuuuuchen/Active-Contour-Loss/blob/master/Active-Contour-Loss.py"""


import torch as torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def Active_Contour_Loss(y_true, y_pred): 

	"""
	lenth term
	"""

	x = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal and vertical directions 
	y = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1]

	delta_x = x[:,:,1:,:-2]**2
	delta_y = y[:,:,:-2,1:]**2
	delta_u = torch.abs(delta_x + delta_y) 

	lenth = torch.mean(torch.sqrt(delta_u + 0.00000001)) # equ.(11) in the paper

	"""
	region term
	"""

	C_1 = torch.ones_like(y_true[0,0])
	C_2 = torch.zeros_like(y_true[0,0])

	region_in = torch.abs(torch.mean( y_pred[:,0,:,:] * ((y_true[:,0,:,:] - C_1)**2) ) ) # equ.(12) in the paper
	region_out = torch.abs(torch.mean( (1-y_pred[:,0,:,:]) * ((y_true[:,0,:,:] - C_2)**2) )) # equ.(12) in the paper

	lambdaP = 1 # lambda parameter could be various.
	mu = 1 # mu parameter could be various.
	
	return lenth + lambdaP * (mu * region_in + region_out) 