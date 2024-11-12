import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import json
import os
import pandas as pd

style.use('dark_background')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def load_params():
    with open("SGSC_params.json", "r") as f: 
        return json.load(f)

def animate(i):
    params = load_params()

    output_results_to_track = f"output/{params['output_dir']}/train_output.csv"
    
    ax1.clear()
    
    try:
        df = pd.read_csv(output_results_to_track)

        ax1.plot([0] + df.get("accuracy").tolist())
        
        #ax1.annotate("high point", (max([0] + df.get("accuracy").tolist()), 1))
        ax1.annotate('Local Max', xy =(3.3, 1), 
                    xytext =(3, 1.8),  
                    arrowprops = dict(facecolor ='blue', 
                                    shrink = 0.05),) 
        ax1.annotate(f"Highest Accuracy of {round(max([0] + df.get('accuracy').tolist()), 3) * 100}%", 
                    (df.get("accuracy").tolist().index(max(df.get("accuracy").tolist())) + 1,
                    max([0] + df.get("accuracy").tolist())), 
                    xytext=(len(df.get("accuracy").tolist())/2, 0.1), 
                    arrowprops=dict(arrowstyle='-|>'))
        
        print(max([0] + df.get("accuracy").tolist()), df.get("accuracy").tolist().index(max(df.get("accuracy").tolist())))
        
    except:
        print("csv empty, will try again in 5 seconds!")    
        
    ax1.set_title(f"accuracy of model run: {params['output_dir']}")
    ax1.set_ylabel("accuracy (%)")
    ax1.set_xlabel(f"epoch 0 - {params['NUM_EPOCH']}")
    
ani = animation.FuncAnimation(fig, animate, interval=20000)

#plt.tight_layout()
plt.show()



"""
df = pd.read_csv(output_results_to_track)

plt.plot(df.get("accuracy").tolist())
plt.show()
"""