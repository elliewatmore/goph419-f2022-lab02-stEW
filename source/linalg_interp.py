import numpy as np

def gauss_iter_solve(A,b,x0=None, tol = 1e-8, alg = 'seidel'):
	"""
	"""
	A = np.array(A,dtype = float)
	b = np.array(b, dtype = float)
	if (ndimA := len(A.shape)) != 2:
		raise ValueError("A is {ndimA}-dimensional but it should be 2D")
	if not (n := A.shape[0]) == (m := A.shape[1]):
		raise ValueError("A has {n} rows and {m} columns but it should be square, n=m")
	if (ndimb := len(b.shape)) not in [1, 2]:
		raise ValueError("b is {ndimb}-dimensional but it should be either 1D or 2D")

	nb = b.shape[0]
	if rhs_1d := (ndimb == 1): b = np.reshape(b, (nb,mb :=1))
	else: mb=b.shape[1]
	if (mb := b.shape[0]) != n:
		raise ValueError("b has {mb} rows but should have the same number of rows as A, which has {n} rows")
	

	if x0 is None:
		x0 = np.zeros((b.shape))
	else:
		if (ndimx := len(x0.shape))==1:
			x0 = np.reshape(x0, (n, 1))  
		if not rhs_1d and x0.shape[1] == 1:
			x0 = x0 @ np.ones((1, mb))

	alg = alg.strip().lower()
	if alg not in ['seidel','jacobi']:
		raise ValueError('alg contains a string that is neither jacobi or seidel')
	

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
		if alg == 'jacobi':
			x_old = np.array(x0)
			x0  = B_star - (A_0_star @ x0)
		elif alg == 'seidel':
			x_old = np.array(x0)
			for i, _ in enumerate(A):
				x0[i,:] = B_star[i:i+1,:] - A_0_star[i:i+1, :] @ x0[:,:]
		n_iter += 1
		dx = x0 - x_old
		eps_a = np.linalg.norm(dx, axis = 0) / np.linalg.norm(x0, axis = 0)
	
	x0=np.reshape(x0,(n,))
	return x0


def spline_function(xd,yd, order = 3):
	"""
	"""
	xd = np.array(xd, dtype = float)
	yd = np.array(yd, dtype = float)
	if len(xd) != len(yd):
		raise ValueError('the number of independant variables must equal the number of dependant variables')
	xduniq = np.unique(xd)
	if len(xduniq) != len(xd):
		raise ValueError('the array xd contains repeated values')
	if (all (xd[i] >= xd[i+1] for i in range(len(xd)-1))):
		raise ValueError('the xd values are not in increaseing order')
	dx = np.diff(xd)
	dy = np.diff(yd)
	df1 = dy/dx
	if order == 1:
		def s1(x):
			a = yd[:-1]
			b = df1[:-1]
			for xk in x:
				k = np.array([np.nonzero(xd>=xk)[0][0]-1 for xk in x])
				k = np.where(k<0,0,k)
				y = a[k-1] + b[k-1] * (x-xd[k-1])
			return y
		return s1
	elif order == 2:
		def s2(x):
			N = xd.shape[0]
			rhs = np.zeros(N-1)
			rhs[1:] = np.diff(df1, axis=0)
			N = len(xd)
			A = np.zeros((N-1,N-1))
			A[0,0:2] = [1, -1] 
			A[1:,:-1] += np.diag(dx[:-1])
			A[1:,1:] += np.diag(dx[1:])
			c = np.linalg.solve(A,rhs)
			b = dy - (c * dx)
			a = yd[:-1]
			for xk in x:
				k = np.array([np.nonzero(xd>=xk)[0][0]-1 for xk in x])
				k = np.where(k<0,0,k)
				y = a[k] + b[k] * (x-xd[k]) + c[k] * (x - xd[k])**2
			return y
		return s2
	elif order ==3:
		def s3(x):
			N = xd.shape[0] # finding length of input data array
			diff_matrix = np.diff(df1)
			rhs = np.zeros(N)
			rhs[1:-1] = 3 * diff_matrix
			N = len(xd)
			A = np.zeros((N,N))
			A[1,0] = dx[0]
			A[-2,-1] = dx[-1]
			A[0,:3] = [-dx[1], (dx[0]+dx[1]), -dx[0]]
			A[-1,-3:] = [-dx[-1], (dx[-1]+dx[-2]), -dx[-2]]
			A[1:-1,:-2] += np.diag(dx[:-1])
			A[1:-1,1:-1] +=np.diag(2*(dx[:-1]+dx[1:]))
			A[1:-1,2:] += np.diag(dx[1:])
			c = np.linalg.solve(A,rhs)
			d = np.diff(c)/(dx *3)
			b = df1 - dx * ( c[:-1] + c[1:] * 2)/3
			k = np.array([np.nonzero(xd>=xk)[0][0]-1 for xk in x])
			k = np.where(k<0,0,k)
			y = np.array([(yd[k]
						+ b[k] * (xk -xd[k])
						+ c[k] * (xk - xd[k])**2
						+ d[k] * (xk - xd[k])**3)
						for k, xk in zip(k, x)])
			return y
		return s3
	else:
		raise ValueError('order is not 1, 2, or 3')


