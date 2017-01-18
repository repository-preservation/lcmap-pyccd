**RobustFit ( Ordinary Least Squares - OLS)**  

This is a standard python module.  
( Robust Linear Models may be investigated )  

Inputs: dates, dependent values  

num years = ceiling ( ( beginning date – end date ) / 365.25 )  

W1 = 2 * pi / 365.25  
W2 = W1 / num years  

independent_arr [ column1 ] = 1’s  
independent_arr [ column2 ] = [ cos ( W1 * d ) for d in dates ]  
independent_arr [ column3 ] = [ sin ( W1 * d ) for d in dates ]  
independent_arr [ column4 ] = [ cos ( W2 * d ) for d in dates ]  
independent_arr [ column5 ] = [ sin ( W2 * d ) for d in dates ]  

regress  

return coefficients  

note: Number of years refers to the span of years represented by the model  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;window, rounded up
