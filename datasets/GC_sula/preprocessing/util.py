import numpy as np
from scipy import signal


def lowpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2                           
    wp = fp / fn                                  
    ws = fs / fn                                  
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  
    b, a = signal.butter(N, Wn, "low")            
    y = signal.filtfilt(b, a, x)

    return y


def deg2meter(data_deg):
    data_meter = data_deg * 111111 # 111111 [m] = 40000[km] / 360[deg] * 1000
    
    return data_meter


def get_pixel_meter_ratio(px1, px2, px3, px4, px5, px6, px7, px8, number):

    px = []
    px.extend(px1)
    px.extend(px2)
    px.extend(px3)
    px.extend(px4)
    px.extend(px5)
    px.extend(px6)
    px.extend(px7)
    px.extend(px8)

    x_min = np.min(px)
    x_max = np.max(px)
    x_diff = x_max - x_min

    # estimated diameter ratio between the movable range and the container (depend on camera position)
    if number == 1:
        diameter_ratio = 0.685
    elif number == 2:
        diameter_ratio = 0.690 

    container_diameter_pixel = x_diff / diameter_ratio
    container_diameter_meter = 0.113 #11.3[cm]
    
    ratio = container_diameter_meter / container_diameter_pixel
    
    return ratio
        

def pixel2meter(data_pixel, ratio):
    data_meter = data_pixel * ratio
    
    return data_meter


def outlier2nan(dpx_a, dpy_a, dpx_b, dpy_b, dpx_c, dpy_c, px_a, py_a, px_b, py_b, px_c, py_c, number):
    
    if number == 1:
        min_x, max_x, min_y, max_y, time_window = 50, 650, 50, 750, 6
    elif number == 2:
        min_x, max_x, min_y, max_y, time_window = 0, 800, 100, 900, 6
    elif number == 3:
        min_x, max_x, min_y, max_y, time_window = 30, 700, 20, 770, 6
        
    px_a = px_a
    py_a = py_a
    px_b = px_b
    py_b = py_b
    px_c = px_c
    py_c = py_c
    
    dpxs = np.hstack((dpx_a,dpx_b,dpx_c))
    mean_dpxs = np.mean(dpxs)
    std_dpxs = np.std(dpxs)
    s3 = std_dpxs * 3

    for i in range(len(dpx_a)):
        px_ai = px_a[i]
        py_ai = py_a[i]
        px_bi = px_b[i]
        py_bi = py_b[i]
        px_ci = px_c[i]
        py_ci = py_c[i]

        dpx_ai = dpx_a[i]
        dpy_ai = dpy_a[i]
        dpx_bi = dpx_b[i]
        dpy_bi = dpy_b[i]
        dpx_ci = dpx_c[i]
        dpy_ci = dpy_c[i]

        if dpx_ai < -s3 or dpx_ai > s3 or dpy_ai < -s3 or dpy_ai > s3 or dpx_bi < -s3 \
            or dpx_bi > s3 or dpy_bi < -s3 or dpy_bi > s3 or dpx_ci < -s3 or dpx_ci > s3 or dpy_ci < -s3 or dpy_ci > s3 \
            or px_ai < min_x or px_ai > max_x or px_bi < min_x or px_bi > max_x or px_ci < min_x or px_ci > max_x \
            or py_ai < min_y or py_ai > max_y or py_bi < min_y or py_bi > max_y or py_ci < min_y or py_ci > max_y:
            px_a[i:i+time_window+1] = np.nan
            py_a[i:i+time_window+1] = np.nan
            px_b[i:i+time_window+1] = np.nan
            py_b[i:i+time_window+1] = np.nan
            px_c[i:i+time_window+1] = np.nan
            py_c[i:i+time_window+1] = np.nan
    
    return px_a, py_a, px_b, py_b, px_c, py_c
    
    
def normalization(px1, py1, px2, py2, px3, py3, number):
    
    py1 = py1 * (-1)
    py2 = py2 * (-1)
    py3 = py3 * (-1)
    
    if number == 1:
        x_low, x_range, y_low, y_range = 40, 610, -750, 670
    elif number == 2:
        x_low, x_range, y_low, y_range = 77, 623, -820, 689
    elif number == 3:
        x_low, x_range, y_low, y_range = 60, 630, -745, 690
                
    # Actual container size: 55 * 60 cm
    px1 = (px1 - x_low) / x_range * 550 / 1000
    py1 = (py1 - y_low) / y_range * 600 / 1000
    px2 = (px2 - x_low) / x_range * 550 / 1000
    py2 = (py2 - y_low) / y_range * 600 / 1000
    px3 = (px3 - x_low) / x_range * 550 / 1000
    py3 = (py3 - y_low) / y_range * 600 / 1000
    
    return px1, py1, px2, py2, px3, py3
    