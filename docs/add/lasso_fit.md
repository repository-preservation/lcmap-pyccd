**LASSOFit**  
Inputs: dates, dependent values, degrees of freedom (df)  

W = 2 * pi / 365.25  

Independent_arr[column1] = dates  

If df >= 4:  
&nbsp;&nbsp;&nbsp;&nbsp;Independent_arr[column2] = cos(W * d) for d in dates  
&nbsp;&nbsp;&nbsp;&nbsp;Independent_arr[column3] = sin(W * d) for d in dates  

If df >= 6:  
&nbsp;&nbsp;&nbsp;&nbsp;Independent_arr[column4] = cos(2 * W * d) for d in dates  
&nbsp;&nbsp;&nbsp;&nbsp;Independent_arr[column5] = sin(2 * W * d) for d in dates  

If df >= 8:  
&nbsp;&nbsp;&nbsp;&nbsp;Independent_arr[column6] = cos(3 * W * d) for d in dates  
&nbsp;&nbsp;&nbsp;&nbsp;Independent_arr[column7] = sin(3 * W * d) for d in dates  

Regress  

Return fitted coefficients, RMSE, residuals (actual - predicted)  
