import sympy as s
x = s.Symbol('x')
usgb = 1.3
desired_income = 70
gb_tax = 0.4
us_fl_tax = 0.22
us_ca_tax = 0.45
us_wa_tax = 0.25
ua_tax = 0.05
# salary * tax - desired_income - (apartment + food/transport) * 12
print('UK', s.solve(x * (1 - gb_tax) - desired_income / usgb - (2 + 1) * 12, x))
print('US_FL', s.solve(x * (1 - us_fl_tax) - desired_income - (1.2 + 0.7) * 12, x))
print('US_CA', s.solve(x * (1 - us_ca_tax) - desired_income - (3 + 1) * 12, x))
print('US_LA', s.solve(x * (1 - us_ca_tax) - desired_income - (1.8 + 1) * 12, x))
print('US_WA', s.solve(x * (1 - us_wa_tax) - desired_income - (1.6 + 1) * 12, x))
print('UA, month', s.solve(x * 12 * (1 - ua_tax) - desired_income - (0.05 + 0.3) * 12, x))