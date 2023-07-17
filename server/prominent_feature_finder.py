from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from rdp import rdp
import math
import time


def line_dists(points, start, end):
    if np.all(start == end):
        return np.linalg.norm(points - start, axis=1)

    vec = end - start
    cross = np.cross(vec, start - points)
    return np.divide(abs(cross), np.linalg.norm(vec))


def rdp(M, epsilon=0):
    M = np.array(M)
    start, end = M[0], M[-1]
    dists = line_dists(M, start, end)

    index = np.argmax(dists)
    dmax = dists[index]

    if dmax > epsilon:
        result1 = rdp(M[:index + 1], epsilon)
        result2 = rdp(M[index:], epsilon)

        result = np.vstack((result1[:-1], result2))
    else:
        result = np.array([start, end])

    return result

def normalize_data(data, data_range, chart_size):
    chart_diag = (chart_size['x'] ** 2 + chart_size['y'] ** 2) ** 0.5
    normalize_function = lambda axis: lambda d_x: (d_x - data_range[axis]['min']) / (data_range[axis]['max'] - data_range[axis]['min']) * chart_size[axis] / chart_diag
    return {'x': list(map(normalize_function('x'), data['date'])), 'y': list(map(normalize_function('y'), data['value']))}

def _get_prominent_features(_data, metadata):
    EPSILON_MAX = 0.25
    EPSILON_STEP = 0.02

    data = {'date': [], 'value': []}
    for _datum in _data:
        data['date'].append(_datum['date'])
        data['value'].append(_datum['value'])

    data_range = metadata['range']
    data_range['x']['min'] = datetime.strptime(data_range['x']['min'], '%Y-%m-%d')
    data_range['x']['max'] = datetime.strptime(data_range['x']['max'], '%Y-%m-%d')
    chart_size = metadata['size']

    data_n = normalize_data(data, data_range, chart_size)

    normalization_factor = (chart_size['x'] ** 2 + chart_size['y'] ** 2) ** 0.5
    chart_range = {'x': {'min': 0,'max': chart_size['x'] / normalization_factor}, 'y': {'min': 0, 'max': chart_size['y'] / normalization_factor}}

    epsilons = np.arange(EPSILON_MAX, 0, -EPSILON_STEP)

    simplify_start_time = time.time()
    print(f'#data points: {len(data_n["x"])}')
    data_n2 = {}
    if len(data_n['x']) > 50:
        data_s0 = rdp(list(zip(data_n['x'], data_n['y'])), epsilon = 0.02)
        data_n2['x'], data_n2['y'] = zip(*data_s0)
    else:
        data_n2 = data_n
    print(f'#simplified data points: {len(data_n2["x"])}')
    # data_n2 = data_n

    # Get point features
    point_features = {}
    max_x = -1
    max_idx = None
    for epsilon in epsilons:
        data_s = rdp(list(zip(data_n2['x'], data_n2['y'])), epsilon = epsilon)
        sx, sy = zip(*data_s)
        for i, x in enumerate(sx):
            if x not in point_features:
                front_slope = None
                back_slope = None
                point_type = None
                point_subtype = None
                if i > 0:
                    front_slope = sy[i] - sy[i - 1]
                if i + 1 < len(sx):
                    back_slope = sy[i + 1] - sy[i]
                if front_slope == None:
                    point_type = "extremum"
                    if back_slope > 0:
                        point_subtype = "-"
                    elif back_slope < 0:
                        point_subtype = "+"
                    else:
                        point_type = "inflection"
                elif back_slope == None:
                    point_type = "extremum"
                    if front_slope > 0:
                        point_subtype = "+"
                    elif front_slope < 0:
                        point_subtype = "-"
                    else:
                        point_type = "inflection"
                else:
                    if front_slope > 0 and back_slope < 0:
                        point_type = "extremum"
                        point_subtype = "+"
                    elif front_slope < 0 and back_slope > 0:
                        point_type = "extremum"
                        point_subtype = "-"
                    else:
                        point_type = "inflection"
                type_tag = f"{point_type}:"
                if point_subtype != None:
                    type_tag += point_subtype
                point_features[x] = {'start': epsilon, 'end': EPSILON_STEP, 'persistence': epsilon - EPSILON_STEP, 'location': {'x': x, 'y': sy[i]}, 'type': type_tag}
                if epsilon == EPSILON_MAX and max_x < x:
                    max_x = x
                    max_idx = len(point_features) - 1

    point_features = list(point_features.values())
    point_features.sort(key = lambda d: d['persistence'] * 100 - d['location']['x'], reverse = True)

    # Get trend features
    trend_features = []
    live_trends = []
    init_flag = True
    prev_point = None
    for point_feature in point_features:
        if init_flag:
            if math.isclose(point_feature['start'], EPSILON_MAX): 
                if prev_point != None:
                    live_trends[-1]['end_point'] = point_feature['location']
                    live_trends[-1]['angle'] = math.atan((live_trends[-1]['end_point']['y'] - live_trends[-1]['start_point']['y']) / (live_trends[-1]['end_point']['x'] - live_trends[-1]['start_point']['x']))
                    trend_subtype = "0"
                    if live_trends[-1]['angle'] > 0:
                        trend_subtype = "+"
                    elif live_trends[-1]['angle'] < 0:
                        trend_subtype = "-"
                    live_trends[-1]['type'] = f"trend:{trend_subtype}"
                live_trends.append({'start_point': point_feature['location'], 'end_point': None, 'start': EPSILON_MAX, 'end': None, 'persistence': None, 'effective': True, 'parent': None, 'angle': None})
                prev_point = point_feature['location']['x']
                continue
            else:
                live_trends = live_trends[:-1]
                init_flag = False
        for i, live_trend in enumerate(live_trends):
            if live_trend['start_point']['x'] < point_feature['location']['x'] and live_trend['end_point']['x'] > point_feature['location']['x']:
                live_trend['end'] = point_feature['start']
                live_trend['persistence'] = live_trend['start'] - live_trend['end']
                trend_features.append(live_trend)

                earlier_trend = {'start_point': live_trend['start_point'], 'end_point': point_feature['location'], 'start': point_feature['start'], 'end': None, 'persistence': None, 'effective': True, 'parent': live_trend}
                earlier_trend['angle'] = math.atan((earlier_trend['end_point']['y'] - earlier_trend['start_point']['y']) / (earlier_trend['end_point']['x'] - earlier_trend['start_point']['x']))
                trend_subtype = "0"
                if earlier_trend['angle'] > 0:
                    trend_subtype = "+"
                elif earlier_trend['angle'] < 0:
                    trend_subtype = "-"
                earlier_trend['type'] = f"trend:{trend_subtype}"
                
                latter_trend = {'start_point': point_feature['location'], 'end_point': live_trend['end_point'], 'start': point_feature['start'], 'end': None, 'persistence': None, 'effective': True, 'parent': live_trend}
                latter_trend['angle'] = math.atan((latter_trend['end_point']['y'] - latter_trend['start_point']['y']) / (latter_trend['end_point']['x'] - latter_trend['start_point']['x']))
                trend_subtype = "0"
                if latter_trend['angle'] > 0:
                    trend_subtype = "+"
                elif latter_trend['angle'] < 0:
                    trend_subtype = "-"
                latter_trend['type'] = f"trend:{trend_subtype}"
                

                live_trends[i : i + 1] = earlier_trend, latter_trend
                break

    for live_trend in live_trends:
        live_trend['end'] = EPSILON_STEP
        live_trend['persistence'] = live_trend['start'] - live_trend['end']
        trend_features.append(live_trend)

    trend_features.sort(key = lambda d: d['persistence'], reverse = True)

    prominent_points = point_features[:]
    prominent_points.pop(max_idx)
    prominent_points = prominent_points[1:]
    features = prominent_points + trend_features
    features.sort(key = lambda d: d['persistence'], reverse = True)
    simplify_end_time = time.time()
    print(f'simplification time: {simplify_end_time - simplify_start_time}s')

    search_start_time = time.time()
    features_cleaned = []
    for feature in features:
        if 'start_point' in feature:
            for i, d in enumerate(data_n['x']):
                if abs(d - feature['start_point']['x']) < 1e-8:
                    start_idx = i
                if abs(d - feature['end_point']['x']) < 1e-8:
                    end_idx = i
                    break
            feature_dates = [data['date'][start_idx].strftime("%Y-%m-%d"), data['date'][end_idx].strftime("%Y-%m-%d")]
            prominence = feature['persistence']
            feature_type = feature['type']
        else:
            for i, d in enumerate(data_n['x']):
                if abs(d - feature['location']['x']) < 1e-8:
                    location_idx = i
                    break
            feature_dates = data['date'][location_idx].strftime("%Y-%m-%d")
            prominence = feature['persistence']
            feature_type = feature['type']
        features_cleaned.append({'dates': feature_dates, 'prominence': prominence, 'type': feature_type})
    search_end_time = time.time()
    print(f'search time: {search_end_time - search_start_time}s')
    return features_cleaned


# Generate figures
# plt.figure()
# plt.xlim([0, len(prominence) + 1])
# plt.ylim([0, 0.25])
# plt.scatter(range(1, len(prominence) + 1), prominence, facecolors='none', edgecolors='gray')#c = 'gray', marker = 'o')

# colors = ['red', 'green', 'blue']
# for kim_feature_idx, kim_feature in enumerate(kim_features):
#     if "ours1" in kim_feature[2]:
#         plt.scatter([1], [prominence[0]], c = colors[kim_feature_idx])
#     elif "ours2" in kim_feature[2]:
#         plt.scatter([2], [prominence[1]], c = colors[kim_feature_idx])
#     elif "ours3" in kim_feature[2]:
#         plt.scatter([3], [prominence[2]], c = colors[kim_feature_idx])
#     else:
#         for feature_idx, feature in enumerate(features):
#             if feature_idx < 3:
#                 continue
#             if feature_idx >= len(prominence):
#                 break
#             if isinstance(kim_feature[0], list) and 'start_point' in feature:
# #                     print(feature['start_point'])
# #                     print(data['date'])
# #                     print(data_n['x'][data['date'].index(datetime.strptime(kim_feature[0][0], '%Y-%m-%d'))])
# #                     throw
#                 if data_n['x'][data['date'].index(datetime.strptime(kim_feature[0][0], '%Y-%m-%d'))] == feature['start_point']['x'] and data_n['x'][data['date'].index(datetime.strptime(kim_feature[0][1], '%Y-%m-%d'))] == feature['end_point']['x']:
#                     plt.scatter([feature_idx + 1], [prominence[feature_idx]], c = colors[kim_feature_idx])
#                     break
#             elif not(isinstance(kim_feature[0], list)) and 'start_point' not in feature:
#                 if data_n['x'][data['date'].index(datetime.strptime(kim_feature[0], '%Y-%m-%d'))] == feature['location']['x']:
#                     print("HEY")
#                     plt.scatter([feature_idx + 1], [prominence[feature_idx]], c = colors[kim_feature_idx])
#                     break