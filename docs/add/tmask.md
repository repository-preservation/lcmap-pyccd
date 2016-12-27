**Tmask Function**  
Inputs: dates, band 2, band 5, median variogram value band 2, median variogram value band 5, T_Const  

[RobustFit] (robust_fit.md) band 2  
[RobustFit] (robust_fit.md) band 5  

Delta band 2 = observed values – predicted values  
Delta band 5 = observed values – predicted values  

Tmask value = 1  
&nbsp;&nbsp;&nbsp;&nbsp;Where (Delta band 2 > median variogram band 2 * T_Const)  
&nbsp;&nbsp;&nbsp;&nbsp;Or    (Delta band 5 > median variogram band 5 * T_Const)  

else Tmask value = 0  

return Tmask  
