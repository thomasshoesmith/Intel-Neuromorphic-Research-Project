from SGSC_lava_inference_Loihi_plot_script import Lava_Run

scan_combinations_2=[(30, 2.0),
                     (28, 2.0),
                     (26, 2.0),
                     (24, 2.0),
                     (22, 2.0),
                     (20, 2.0),
                     (18, 2.0),
                     (16, 2.0),
                     (14, 2.0),
                     (12, 2.0),
                     (10, 2.0),
                     (8, 2.0),
                     (6, 2.0),
                     (4, 2.0),
                     (2, 2.0),
                     (1, 2.0)]

scan_combinations_4=[(30, 4.0),
                     (28, 4.0),
                     (26, 4.0),
                     (24, 4.0),
                     (22, 4.0),
                     (20, 4.0),
                     (18, 4.0),
                     (16, 4.0),
                     (14, 4.0),
                     (12, 4.0),
                     (10, 4.0),
                     (8, 4.0),
                     (6, 4.0),
                     (4, 4.0),
                     (2, 4.0),
                     (1, 4.0)]

scan_combinations_8=[(30, 8.0),
                     (28, 8.0),
                     (26, 8.0),
                     (24, 8.0),
                     (22, 8.0),
                     (20, 8.0),
                     (18, 8.0),
                     (16, 8.0),
                     (14, 8.0),
                     (12, 8.0),
                     (10, 8.0),
                     (8, 8.0),
                     (6, 8.0),
                     (4, 8.0),
                     (2, 8.0),
                     (1, 8.0)]

scan_combinations_16=[(30, 16.0),
                     (28, 16.0),
                     (26, 16.0),
                     (24, 16.0),
                     (22, 16.0),
                     (20, 16.0),
                     (18, 16.0),
                     (16, 16.0),
                     (14, 16.0),
                     (12, 16.0),
                     (10, 16.0),
                     (8, 16.0),
                     (6, 16.0),
                     (4, 16.0),
                     (2, 16.0),
                     (1, 16.0)]

scan_combinations = scan_combinations_2 + scan_combinations_4 + scan_combinations_8 + scan_combinations_16

for scan in scan_combinations:
    try:
        Lava_Run(scale_val = scan[1], vthres = scan[0])
    except RuntimeError:
        print(f"failed run at scale_val = {scan[1]} and vthres = {scan[0]}")    