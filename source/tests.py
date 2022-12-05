import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from linalg_interp import spline_function
from linalg_interp import gauss_iter_solve

#testing gauss iteration function

#testing jacobi with an inital guess
A = np.array([[9,1,4],[2,7,1],[3,2,6]])
b = np.array([11,13,12])
guess = np.array([1,1,1])

sol = gauss_iter_solve(A,b, x0 =guess, alg='jacobi')

print('The solution using jacobi with an initial guess is',sol)

solution = np.linalg.solve(A,b)
print('The solution using built in algorithm is', solution)
print(' ')

#testing jacobi with no inital guess 

A = np.array([[15,2,-4],[-1,17,1],[2,5,20]])
b = np.array([6,-2,10])

sol = gauss_iter_solve(A,b, alg='jacobi')

print('The solution using jacobi with no initial guess is',sol)

solution = np.linalg.solve(A,b)
print('The solution using built in algorithm is', solution)
print ('')

#testing seidel with no inital guess
A = np.array([[9,1,4],[2,7,1],[3,2,6]])
b = np.array([11,13,12])

sol = gauss_iter_solve(A,b)

print('The solution using seidel with no inital guess is',sol)

solution = np.linalg.solve(A,b)
print('The solution using built in algorithm is', solution)
print ('')

# testing seidel with an inital guess
A = np.array([[15,2,-4],[-1,17,1],[2,5,20]])
b = np.array([6,-2,10])
guess = np.array([1,1,1])


sol = gauss_iter_solve(A,b, x0 = guess, alg='jacobi')

print('The solution using seidel with an initial guess is',sol)

solution = np.linalg.solve(A,b)
print('The solution using built in algorithm is', solution)
print('')


# test for rhs matrix such that result is inverse of A

#testing seidel 
A = ([[11,2,4],[3,9,2],[1,1,13]])
b = np.eye(len(A))
x = gauss_iter_solve(A, b)
print('solution using seidel is', x)
solution = np.linalg.solve(A,b)
print('solution using built in algorith is',solution)
ident = A @ x
print('A @ x =', ident)
print('')

#testing jacobi
A = ([[45,5,10],[6,29,4],[10,7,30]])
b = np.eye(len(A))
x = gauss_iter_solve(A, b, alg ='jacobi')
print('solution using jacobi is', x)
solution = np.linalg.solve(A,b)
print('solution using built in algorithm is',solution)
ident = A @ x
print('A @ x =', ident)
print('')



# spline function tests
# unit tests from linear, quadratic, and cubic function 


xd = np.linspace(-10,10,15)

#linear fucntion
yd1 =  2*xd
#solving for spline interpolation functions
spline1 = spline_function(xd,yd1, order =1)  
#calculating spline data     
y1 = spline1(xd)


#quadratic function
yd2 = 0.5*(xd**2)
#solving for spline interpolation functions
spline2 = spline_function(xd, yd2, order =2)
#calcualting spline data
y2 = spline2(xd)

#cubic function
yd3 = 0.05*(xd**3)
# solving for spline interpolation functions
spline3 = spline_function(xd, yd3, order =3)
#calculating spline data
y3 = spline3(xd)



# plotting data points for linear, quadratic and cubic spline functions
plt.plot(xd,yd1, 'ro')
plt.plot(xd,y1,'m', label = 'linear function')

plt.plot(xd,yd2, 'bs')
plt.plot(xd,y2,'c', label = 'quadratic function')

plt.plot(xd,yd3, 'g^')
plt.plot(xd,y3,'tab:green', label = 'cubic function')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Spline interpolation for linear, quadratic and cubic functions')
plt.legend()
plt.show()


# unit test to compare results with scipy.interpolate

#testing for exponential function
yd4 = np.exp(xd)
#solving for spline interpolation functions
spline4 = spline_function(xd, yd4, order=3)
# calculating interpolated spline data
y4 = spline4(xd)

#calculating spline function and data using built in function
spline_u = UnivariateSpline(xd, yd4, k=3, s=0)
y_u = spline_u(xd)

# plotting spline data for my function and built in function
plt.plot(xd,yd4, 'ro')
plt.plot(xd,y4,'m', label = 'my spline function')
plt.plot(xd,y_u,'c', label = 'Univariate spline function')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of my spline function and Univariate spline function for exponential function')
plt.legend()
plt.show()