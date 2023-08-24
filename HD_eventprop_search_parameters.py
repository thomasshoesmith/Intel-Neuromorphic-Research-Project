# script to generate various parameters to sweep 

hidden_w_mean = []
hidden_w_sd = []
output_w_mean = []
output_w_sd = []

no_of_val = 11
div = 2.5

for i in range(no_of_val):
    hidden_w_mean.append(i/div)
    hidden_w_sd.append(i/div)
    output_w_mean.append(i/div)
    output_w_sd.append(i/div)

combinations = []

print(hidden_w_mean)

for hwm in hidden_w_mean:
    for hwsd in hidden_w_sd:
        for owm in output_w_mean:
            for owsd in output_w_sd:
                combinations.append([hwm, hwsd, owm, owsd])
                
print(len(combinations))