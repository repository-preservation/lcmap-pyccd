**RobustFit (OLS or IRLS?)**  
Inputs: dates, dependent values  

Num years = ceiling((beginning date – end date) / 365.25)  

W1 = 2 * pi / 365.25  
W2 = W1 / num years  

Independent_arr[column1] = 1’s  
Independent_arr[column2] = [cos(W1 * d) for d in dates]  
Independent_arr[column3] = [sin(W1 * d) for d in dates]  
Independent_arr[column4] = [cos(W2 * d) for d in dates]  
Independent_arr[column5] = [sin(W2 * d) for d in dates]  

Regress  

Return coefficients  
