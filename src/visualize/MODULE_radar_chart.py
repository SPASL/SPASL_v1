import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from collections import Counter

def draw_radar_chart_1_network_SPASL_v1(network_name:str, acc_1:list[float], acc_5:list[float], bm_name = 'SPASL_v1', save_path = None, show_plot = False, show_ticks = False):
    ''' 
        plot a single network on a single benchmark for its Top-1 and Top-5 performance
        network_name = 'net1'
        bm_name = default to 'SPASL_v1'
        acc_1 = [11.11,11.11,11.11,11.11,11.11]
        acc_5 = [11.11,11.11,11.11,11.11,11.11]
        show_ticks = False(default)/True  (Whether to show the 0/20/40/../100 ticks on the radar plot)        
        save_path = './'  --> the filename will be f'{network_name}_{bm_name}_{[acc_s[-1]]}.png'
        show_plot = False(default) / True (for visualization)
    '''

    plt.close('all')
    acc_1 = [acc_1[0]] + list(reversed(acc_1[1:]))
    acc_5 = [acc_5[0]] + list(reversed(acc_5[1:]))
    
    theta2 = radar_factory(5, frame='polygon') 
    corner_labels = ['SH', 'LR', 'SN', 'AA', 'PI']
    
    fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1,
                            subplot_kw=dict(projection='radar'))
    if show_ticks:
        ax.set_rgrids((0,20,40,60, 80,100), labels = ('0   20   40   60   80   100', '', '', '', '', ''))
    else:
        ax.set_rgrids((0,20,40,60, 80,100), labels=('', '', '', '', '', ''))

    ax.set_title(f'{network_name} on {bm_name}\n',weight="bold",  size=18, position=(0.5, 0),
            horizontalalignment='center', verticalalignment='center')

    ax.set_ylim([0,100])
    ax.plot(theta2, acc_1, linewidth=3.0,  color='cornflowerblue')
    ax.fill(theta2, acc_1, facecolor='cornflowerblue', alpha=0.2, label='_nolegend_')
    
    ax.plot(theta2, acc_5, linewidth=3.0,  color='orange')
    ax.fill(theta2, acc_5, facecolor='orange', alpha=0.1, label='_nolegend_')
    
    new_corners = [None for _ in range(5)]
    new_corners[0] = f'{corner_labels[0]}\n{acc_1[0]}  {acc_5[0]}'
    new_corners[1] = f'{corner_labels[1]: <7}\n{acc_1[1]: <10}\n{acc_5[1]: <10}'
    new_corners[2] = f'{corner_labels[2]}\n{acc_1[2]}  {acc_5[2]}'
    new_corners[3] = f'{corner_labels[3]}\n{acc_1[3]}  {acc_5[3]}'
    new_corners[4] = f'{corner_labels[4]: >7}\n{acc_1[4]: >10}\n{acc_5[4]: >10}'
    
    ax.set_varlabels(new_corners)
    
    # add legend relative to top-left plot
    labels = ('Top-1 acc', 'Top-5 acc')
    legend = ax.legend(labels, loc=(0.7, 0.9), labelspacing=0.1, fontsize=13)

    if save_path != None:
        network_name = network_name.replace('/','')
        plt.savefig(f'{save_path}/{network_name}_radar_chart_on_{bm_name}.png', bbox_inches = 'tight')
    
    if show_plot:
        plt.show()



def draw_radar_chart_1_network_1_bm(network_name:str, bm_name:str, acc_1:list[float], acc_5:list[float], save_path = None, show_plot = False, verbose = False, show_ticks = False):
    ''' 
        plot a single network on a single benchmark for its Top-1 and Top-5 performance
        network_name = 'net1'
        bm_name = 'general-10/50/100' / 'resnet-10/50/100' / 'vit-10/50/100'
        acc_1 = [11.11,11.11,11.11,11.11,11.11]
        acc_5 = [11.11,11.11,11.11,11.11,11.11]
        save_path = './'  --> the filename will be f'{network_name}_{bm_name}_{[acc_s[-1]]}.png'
        show_plot = False(default)/True (for visualization)
    '''
    plt.close('all')
    acc_1 = [acc_1[0]] + list(reversed(acc_1[1:]))
    acc_5 = [acc_5[0]] + list(reversed(acc_5[1:]))
    
    theta2 = radar_factory(5, frame='polygon') 
    corner_labels = ['SH', 'LR', 'SN', 'AA', 'PI']
    
    fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1,
                            subplot_kw=dict(projection='radar'))
    if show_ticks:
        ax.set_rgrids((0,20,40,60, 80,100), labels = ('0   20   40   60   80   100', '', '', '', '', ''))
    else:
        ax.set_rgrids((0,20,40,60, 80,100), labels=('', '', '', '', '', ''))
    if verbose:
        ax.set_title(f'{network_name} on SPASL_{bm_name}\n',weight="bold",  size=18, position=(0.5, 0),
                horizontalalignment='center', verticalalignment='center')
    else:
        ax.set_title(f'{network_name}\n',weight="bold",  size=18, position=(0.5, 0),
                horizontalalignment='center', verticalalignment='center')
    ax.set_ylim([0,100])
    ax.plot(theta2, acc_1, linewidth=3.0,  color='cornflowerblue')
    ax.fill(theta2, acc_1, facecolor='cornflowerblue', alpha=0.2, label='_nolegend_')
    
    ax.plot(theta2, acc_5, linewidth=3.0,  color='orange')
    ax.fill(theta2, acc_5, facecolor='orange', alpha=0.1, label='_nolegend_')
    

    
    new_corners = [None for _ in range(5)]
    new_corners[0] = f'{corner_labels[0]}\n{acc_1[0]}  {acc_5[0]}'
    new_corners[1] = f'{corner_labels[1]: <7}\n{acc_1[1]: <10}\n{acc_5[1]: <10}'
    new_corners[2] = f'{corner_labels[2]}\n{acc_1[2]}  {acc_5[2]}'
    new_corners[3] = f'{corner_labels[3]}\n{acc_1[3]}  {acc_5[3]}'
    new_corners[4] = f'{corner_labels[4]: >7}\n{acc_1[4]: >10}\n{acc_5[4]: >10}'
    

    ax.set_varlabels(new_corners)
    
    # add legend relative to top-left plot
    labels = ('Top-1 acc', 'Top-5 acc')
    legend = ax.legend(labels, loc=(0.7, 0.9), labelspacing=0.1, fontsize=13)

    if save_path != None:
        network_name = network_name.replace('/','')
        plt.savefig(f"""{save_path}/{network_name}_SPASL_{bm_name}.png""", bbox_inches = 'tight')
        #print(f'Plot save to {save_path}/{network_name}_SPASL_{bm_name}.png')
    if show_plot:
        plt.show()

def draw_radar_chart_1_network_3_bm(network_name: str, bm_names: list[str], acc_s: list[any],  highlight = None, save_path = None, show_plot = False, verbose = False):
    ''' 
        plot up to 1 network on 3 versions of same bm or same version of 3 bms
        e.g. on vit-10/50/100 and general/
        network_name = 'net1'
        bm_name = 'general-10/50/100' / 'resnet-10/50/100' / 'vit-10/50/100'
        acc_s = [ [11.11,11.11,11.11,11.11,11.11],
                  [11.11,11.11,11.11,11.11,11.11],
                  [11.11,11.11,11.11,11.11,11.11],
                  'Top-1']  # MUST MENTION Top-1 OR Top-5 AS THE LAST ELE IN acc_s
        highlight (if provided) = 'tft' ('t' for True, 'f' for False)
        save_path = './'  --> the filename will be f'{network_name}_on_all_bms_{acc_s[-1]}.png'
        show_plot = False(default)/True (for visualization)
    '''

    if bm_names[0][0] == bm_names[1][0]:
        same_bm = True
        bm_name = bm_names[0].split('-')[0]  # general/resnet/vit
    else:
        same_bm = False
        bm_name = bm_names[0].split('-')[-1]  # 10/50/100
    
    if highlight != None:
        if Counter(highlight)['t'] > 3:
            print('Not enough colors for highlights. At most 3 benchmarks can be highlighted.')
            return
        f_style = ':'
    else:
        highlight = 'f'*3
        f_style = '-'
    plt.close('all')
    
    theta2 = radar_factory(5, frame='polygon') 
    corner_labels = ['SH', 'LR', 'SN', 'AA', 'PI']
    fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1, subplot_kw=dict(projection='radar'))
    ax.set_rgrids((0,20,40,60, 80,100), labels=('', '', '', '', '', ''))
    ax.set_ylim([0,100])

    if verbose:
        if same_bm:
            ax.set_title(f'{network_name} {acc_s[-1]} Acc on SPASL_{bm_name}-10/50/100\n',weight="bold",  size=18, position=(0.5, 0),
                horizontalalignment='center', verticalalignment='center')
        else:
            ax.set_title(f'{network_name} {acc_s[-1]} Acc on all benchmarks-{bm_name}\n',weight="bold",  size=18, position=(0.5, 0),
                horizontalalignment='center', verticalalignment='center')
    else:
        ax.set_title(f'{network_name}\n',weight="bold",  size=18, position=(0.5, 0),
                horizontalalignment='center', verticalalignment='center')
        
    #colors = ['dodgerblue', 'gold','violet', 'forestgreen','darksalmon', 'blueviolet',  'olive', 'teal',  'sandybrown' , 'plum']
    colors = ['#FABB05', '#34A853', '#4285F4' ]
    hightlight_colors = ['red', 'blue', 'lime']
    h_idx = 0
    linestyles = ['-' for _ in range(3)]
    for i in range(3):
        if highlight[i] == 't':
            cur_color = hightlight_colors[h_idx]
            h_idx += 1
        else:
            cur_color = colors[i]
            linestyles[i] = f_style
        #print(acc_s[i])
        acc_s[i] = [acc_s[i][0]] + list(reversed(acc_s[i][1:]))
        #print(acc_s[i])
        ax.plot(theta2, acc_s[i], linewidth=2.5, color=cur_color, linestyle=linestyles[i])
        ax.fill(theta2, acc_s[i], facecolor=cur_color, alpha=0.1, label='_nolegend_')

    new_corners = [None for _ in range(5)]
    new_corners[0] = f'{corner_labels[0]}\n{acc_s[0][0]}  {acc_s[1][0]}  {acc_s[2][0]}'
    new_corners[1] = f'{corner_labels[1]: <7}\n{acc_s[0][1]: <10}\n{acc_s[1][1]: <10}\n{acc_s[2][1]: <10}'
    new_corners[2] = f'{corner_labels[2]}\n{acc_s[0][2]}  {acc_s[1][2]}  {acc_s[2][2]}'
    new_corners[3] = f'{corner_labels[3]}\n{acc_s[0][3]}  {acc_s[1][3]}  {acc_s[2][3]}'
    new_corners[4] = f'{corner_labels[4]: >7}\n{acc_s[0][4]: >10}\n{acc_s[1][4]: >10}\n{acc_s[2][4]: >10}'
    
    
    ax.set_varlabels(new_corners)

    legend = ax.legend(bm_names, loc=(0.8, 0.9), labelspacing=0.02, fontsize=13)

    if save_path != None:
        network_name = network_name.replace('/','')
        if same_bm:
            plt.savefig(f'{save_path}/{network_name}_on_{bm_name}-1050100_{acc_s[-1]}.png', format='png', bbox_inches = 'tight')
        else:
            plt.savefig(f'{save_path}/{network_name}_on_all-{bm_name}_{acc_s[-1]}.png', format='png', bbox_inches = 'tight')
            
    if show_plot:
        plt.show()



def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels, fontsize=15)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':

                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                #return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
                return RegularPolygon((0.5, 0.5), num_vars, radius=.5, facecolor="r")
                
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                #spine.set_bounds(0, 100)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta
