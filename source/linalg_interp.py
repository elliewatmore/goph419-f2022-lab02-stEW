"utility functions"
import numpy as np

def gauss_iter_solve(A,b,x0=None, tol = 1e-8, alg = 'seidel'):
	"""
 	Solve a system A * x = b for x using the Gauss- Siedel iterative approach

    	Parameters
    	----------
    	A : array_like, shape = (n,n)
        	The coefficient matrix.
    	b : array_like, shape = (n,*)
        	The right hand-side vector(s)
    	x0 : array_like, shape = (n,*), or shape = (n,1), optional
        	The initial guess(es), either the same shape as b or a single 
        	column with the same number of rows as A and b. if the latter is used
        	the column will be repeated to initialise the output array with the 
        	same number of columns as b.
        	The default is None, which initialises an output array to a standard g
        	guess of all zeros.
    	tol : float, optional
        	The relative error tolerance (stopping criterion). The default 
        	is 1e-8.
    	alg : string, optional, another possible value of 'jacobi'
        	Indicates whether the classical Gauss-Seidel or Jacobi iteration
        	algorithm should be used. The default is 'seidel'.

    	Returns
    	-------
    	numpy.ndarray, shape = (n,*)
    	The solution to the system with the same shape as b.
    
    	Raises
    	-------
    	ValueError 
        	If alg contains a string other than two options of 'siedel' or 
        	'jacobi', input is case insensitive and ignores leading or trailing 
        	whitespace
    	ValueError 
        	If A is not 2D and square
        	If b is not 1D or 2D, or has a different number of rows from A
        	If x0 is not 1D or 2d, or is different shape from b or has a 
        	different number of rows from A or b
	RuntimeWarning
		If solution does not converge after a specified number of iterations
	"""
	# ensure that A and b are array_like
	A = np.array(A,dtype = float)
	b = np.array(b, dtype = float)

	# make sure A is 2D, if not raise an error
	if (ndimA := len(A.shape)) != 2:
		raise ValueError("A is {ndimA}-dimensional but it should be 2D")
	
	# check that A is square, if not raise an error
	if not (n := A.shape[0]) == (m := A.shape[1]):
		raise ValueError("A has {n} rows and {m} columns but it should be square, n=m")
	
	# make sure b is either 1D or 2D
	if (ndimb := len(b.shape)) not in [1, 2]:
		raise ValueError("b is {ndimb}-dimensional but it should be either 1D or 2D")
	
	
	nb = b.shape[0]
	if rhs_1d := (ndimb == 1): b = np.reshape(b, (nb,mb :=1))
	else: mb=b.shape[1]

	# make sure b has the same number of rows as A
	if (mb := b.shape[0]) != n:
		raise ValueError("b has {mb} rows but should have the same number of rows as A, which has {n} rows")
	
	# initialise x_0, if x_0 is provided initialise with all zeros and reshape
	if x0 is None:
		x0 = np.zeros(n).reshape(-1,1)
	if len(x0.shape)==1: 
		x0 = np.reshape(x0,(n,1))
	if not rhs_1d and x0.shape[1] ==1:
		x0 = x0 @ np.ones((1,mb))

		
	# check algorithm, if neither 'seidel' or 'jacobi' raise an error
	alg = alg.strip().lower()
	if alg not in ['seidel','jacobi']:
		raise ValueError('alg contains a string that is neither jacobi or seidel')
	
	# initialise values
	eps_s = tol
	eps_a = 1
	n_iter = 1 #normalise iterations
	# set number of maximum iterations
	max_it = 100
	# form matrix with only main diagonal entries and zeroes elsewhere
	A_d = np.diag(np.diag(A))
	# calculate inverse of Ad
	A_d_inv = np.linalg.inv(A_d)
	# calculate matrix with only off-diagonal entries and zeroes on main diagonal
	A_0 = A - A_d
	A_0_star = A_d_inv @ A_0
	B_star = A_d_inv @ b
	
	# create a loop to solve Gauss iteration
	while np.max(eps_a) > eps_s and n_iter < max_it:
		# solve using jacobi method
		if alg == 'jacobi':
			x_old = np.array(x0)
			x0  = B_star - (A_0_star @ x0)
		#solve using seidel method
		elif alg == 'seidel':
			x_old = np.array(x0)
			for i, _ in enumerate(A):
				x0[i,:] = B_star[i:i+1,:] - A_0_star[i:i+1, :] @ x0[:,:]
		#increase iteration number by 1
		n_iter += 1
		# calcualte dx to be used to calcualte new approximate error
		dx = x0 - x_old
		# calculate new error
		eps_a = np.linalg.norm(dx, axis = 0) / np.linalg.norm(x0, axis = 0)
		# testing for convergence in specified number of iterations, if system does not converge raise a runtime warning
		if n_iter > max_it:
			raise RuntimeWarning('system does not converge')
	
	#x0=np.reshape(x0,(n,n))
	return x0


def spline_function(xd,yd, order = 3):

	"""
	Generates a spline function given two vectors, x and y, of data

    	Parameters
    	----------
    	xd : array_like float
        	contains data in increasing value
    	yd : array_like float
        	array with same shape as xd
    	order : int, optional
        	contains order of spline function, possible inputs are 1,2, or 3
        	The default is 3
    

    	Returns
    	-------
	function that takes one parameter, float or array_like float, and returns interpolated y value(s)
    	
    
    	Raises
    	-------
    	ValueError 
        	If flattened arrays of xd and yd do not have the same length, such that	
		the number of independant variables does not equal the number of dependant variables
    	ValueError 
        	If there are repeated values in the array xd, so the number of unique 
		independant variable values is not equal to the number of independant 
		values given
	ValueError
		If the values of xd are not in increasing order, the sorted array of 	
		independant variable values is not equal to the flattened array of independant v
		variable values
	
	ValueError
		If the order given is any value other than 1, 2, or 3
	"""
	# ensure xd and yd are array like
	xd = np.array(xd, dtype = float)
	yd = np.array(yd, dtype = float)

	# check that the arrays xd and yd have the same length
	if len(xd) != len(yd):
		raise ValueError('the number of independant variables must equal the number of dependant variables')
	
	#check that there are no repeated values in the array xd
	xduniq = np.unique(xd)
	if len(xduniq) != len(xd):
		raise ValueError('the array xd contains repeated values')

	# make sure values in xd are sorted in increasing order
	if (all (xd[i] >= xd[i+1] for i in range(len(xd)-1))):
		raise ValueError('the xd values are not in increaseing order')
	
	# calcualting differnces for xd and yd
	dx = np.diff(xd)
	dy = np.diff(yd)
	# calculating first order divided difference
	df1 = dy/dx
	
	# create loop to solve for spline function if order is set to 1
	if order == 1:
		#define function
		def s1(x):
			# determines coefficients a and b
			a = yd[:-1]
			b = df1[:-1]
			# for loop to produce spline function
			for xk in x:
				# finding the first index where xd is greater than xk to be used in spline function
				k = np.array([np.nonzero(xd>=xk)[0][0]-1 for xk in x])
				k = np.where(k<0,0,k)
				# defining spline function to be returned
				y = a[k-1] + b[k-1] * (x-xd[k-1])
			return y
		return s1
	# create loop to solve for spline function if order is set to 2
	elif order == 2:
		def s2(x):
			# creating rhs matrix of linear system to be solved
			N = xd.shape[0]
			rhs = np.zeros(N-1)
			rhs[1:] = np.diff(df1, axis=0)
			# creating A matrix of linear system to be solved
			N = len(xd)
			A = np.zeros((N-1,N-1))
			A[0,0:2] = [1, -1] 
			A[1:,:-1] += np.diag(dx[:-1])
			A[1:,1:] += np.diag(dx[1:])
			# finding coefficients a, b, and c
			c = np.linalg.solve(A,rhs)
			b = dy - (c * dx)
			a = yd[:-1]
			# for loop to produce spline function
			for xk in x:
				# finding the first index where xd is greater than xk to be used in spline function
				k = np.array([np.nonzero(xd>=xk)[0][0]-1 for xk in x])
				k = np.where(k<0,0,k)
				#defining spline function to be returned
				y = a[k] + b[k] * (x-xd[k]) + c[k] * (x - xd[k])**2
			return y
		return s2
	# create loop to solve for spline function if order is set to 3
	elif order ==3:
		def s3(x):
			# creating rhs matrix of linear system to be solved
			N = xd.shape[0] # finding length of input data array
			diff_matrix = np.diff(df1)
			rhs = np.zeros(N)
			rhs[1:-1] = 3 * diff_matrix
			# creating A matrix of linear system to be solved
			N = len(xd)
			A = np.zeros((N,N))
			A[1,0] = dx[0]
			A[-2,-1] = dx[-1]
			A[0,:3] = [-dx[1], (dx[0]+dx[1]), -dx[0]]
			A[-1,-3:] = [-dx[-1], (dx[-1]+dx[-2]), -dx[-2]]
			A[1:-1,:-2] += np.diag(dx[:-1])
			A[1:-1,1:-1] +=np.diag(2*(dx[:-1]+dx[1:]))
			A[1:-1,2:] += np.diag(dx[1:])
			# determining coefficients b,c and d
			c = np.linalg.solve(A,rhs)
			d = np.diff(c)/(dx *3)
			b = df1 - dx * ( c[:-1] + c[1:] * 2)/3
			# finding the first index where xd is greater than xk to be used in spline function
			k = np.array([np.nonzero(xd>=xk)[0][0]-1 for xk in x])
			k = np.where(k<0,0,k)
			#defining spline function to be returned
			y = np.array([(yd[k]
						+ b[k] * (xk -xd[k])
						+ c[k] * (xk - xd[k])**2
						+ d[k] * (xk - xd[k])**3)
						for k, xk in zip(k, x)])
			return y
		return s3
	else:
		raise ValueError('order is not 1, 2, or 3')

