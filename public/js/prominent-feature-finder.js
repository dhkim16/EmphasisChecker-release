function _getProminentFeatures(_data, metadata) {
    const EPSILON_MAX = 0.25;
    const EPSILON_STEP = 0.02;

    data_range = metadata['range']
    data_range['x']['min'] = genISODate(data_range['x']['min'])
    data_range['x']['max'] = genISODate(data_range['x']['max'])
    chart_size = metadata['size']

    const data = {'date': [], 'value': []};

    const filteredIndices = _data.date.map((val, i) => (val >= data_range.x.min && val <= data_range.x.max ? i : -1)).filter(val => val !== -1);
    data.date = filteredIndices.map(i => _data.date[i]);
    data.value = filteredIndices.map(i => _data.value[i]);

    const data_n = normalize_data(data, data_range, chart_size)

    const normalization_factor = (chart_size['x'] ** 2 + chart_size['y'] ** 2) ** 0.5
    chart_range = {'x': {'min': 0,'max': chart_size['x'] / normalization_factor}, 'y': {'min': 0, 'max': chart_size['y'] / normalization_factor}}

    const epsilons = Array.from({length: Math.ceil(EPSILON_MAX / EPSILON_STEP)}, (_, i) => EPSILON_MAX - (i * EPSILON_STEP));

    const simplify_start_time = Date.now();
    var data_n2 = {};
    if (data_n['x'].length > 50) {
        const simplifiedPoints = rdp(data_n['x'], data_n['y'], EPSILON_STEP);
        data_n2['x'] = simplifiedPoints[0];
        data_n2['y'] = simplifiedPoints[1];
    } else {
        data_n2 = data_n;
    }
        
    // Get point features
    var point_features = {}
    epsilons.forEach(epsilon => {
        const data_s = rdp(data_n2['x'], data_n2['y'], epsilon)
        const sx = data_s[0]
        const sy = data_s[1]
        sx.forEach((x, i) => {
            if (!(x in point_features)) {
                let front_slope = null
                let back_slope = null
                let point_type = null
                let point_subtype = null
                let type_tag = null
                point_features[x] = {'start': epsilon, 'end': EPSILON_STEP, 'persistence': epsilon - EPSILON_STEP, 'location': {'x': x, 'y': sy[i]}, 'type': type_tag}
            }
        });
    });

    point_features = Object.values(point_features);
    point_features.sort((a, b) => (b['persistence'] * 100 - b['location']['x']) - (a['persistence'] * 100 - a['location']['x']));

    // Get trend features
    const trend_features = []
    var live_trends = []
    var init_flag = true
    var prev_point = null
    for (let pidx = 0; pidx < point_features.length; pidx++) {
        let point_feature = point_features[pidx];
        if (init_flag) {
            if (isClose(point_feature['start'], EPSILON_MAX)) {
                if (prev_point !== null) {
                    live_trends[live_trends.length - 1] ['end_point'] = point_feature['location']
                }
                live_trends.push({'start_point': point_feature['location'], 'end_point': null, 'start': EPSILON_MAX, 'end': null, 'persistence': null, 'effective': true, 'parent': null, 'angle': null})
                prev_point = point_feature['location']['x']
                continue
            } else {
                live_trends = live_trends.slice(0, -1);
                init_flag = false
            }
        }
        for (let i = 0; i < live_trends.length; i++) {
            let live_trend = live_trends[i];
            if (live_trend['start_point']['x'] < point_feature['location']['x'] && live_trend['end_point']['x'] > point_feature['location']['x']) {
                live_trend['end'] = point_feature['start']
                live_trend['persistence'] = live_trend['start'] - live_trend['end']
                trend_features.push(live_trend)

                let earlier_trend = {'start_point': live_trend['start_point'], 'end_point': point_feature['location'], 'start': point_feature['start'], 'end': null, 'persistence': null, 'effective': true, 'parent': live_trend}

                
                let latter_trend = {'start_point': point_feature['location'], 'end_point': live_trend['end_point'], 'start': point_feature['start'], 'end': null, 'persistence': null, 'effective': true, 'parent': live_trend}

                

                live_trends.splice(i, 1, earlier_trend, latter_trend);
                break
            }
        }
    }

    live_trends.forEach(live_trend => {
        live_trend['end'] = EPSILON_STEP
        live_trend['persistence'] = live_trend['start'] - live_trend['end']
        trend_features.push(live_trend);
    });

    if (trend_features.length > 0 && trend_features[trend_features.length - 1].end_point === null) {
        trend_features.splice(-1);
    }
    trend_features.sort((a, b) => b['persistence'] - a['persistence']);

    let prominent_points = point_features.slice();
    if (prominent_points.length === 0) {
        return [];
    }
    let max_persistence = prominent_points[0]['persistence'];
    let pidx;
    for (pidx = 0; pidx < prominent_points.length; pidx++) {   
        let point_feature = prominent_points[pidx];
        if (!isClose(point_feature['persistence'], max_persistence)) {
            break;
        }
    }
    prominent_points.splice(pidx - 1, 1);
    prominent_points = prominent_points.slice(1);
    let features = prominent_points.concat(trend_features);
    features.sort((a, b) => b.persistence - a.persistence);
    const simplify_end_time = Date.now()

    const search_start_time = Date.now()
    features_cleaned = []
    features.forEach(feature => {
        if ('start_point' in feature) {
            let start_idx;
            let end_idx;
            for (let i = 0; i < data_n['x'].length; i++) {
                let d = data_n['x'][i];
                if (isClose(d, feature['start_point']['x'])) {
                    start_idx = i;
                } 
                if (isClose(d, feature['end_point']['x'])) {
                    end_idx = i;
                    break;
                } 
            }
            feature_dates = [data['date'][start_idx].toISOString().slice(0, 10), data['date'][end_idx].toISOString().slice(0, 10)]
            prominence = feature['persistence']
            feature_type = feature['type']
        } else {
            for (let i = 0; i < data_n['x'].length; i++) {
                let d = data_n['x'][i];
                if (isClose(d, feature['location']['x'])) {
                    location_idx = i
                    break
                }
            }
            feature_dates = data['date'][location_idx].toISOString().slice(0, 10)
            prominence = feature['persistence']
            feature_type = feature['type']
        }
        features_cleaned.push({'dates': feature_dates, 'prominence': prominence, 'type': feature_type})
    });
    const search_end_time = Date.now()
    return features_cleaned
}

function normalize_data(data, data_range, chart_size) {
    chart_diag = (chart_size['x'] ** 2 + chart_size['y'] ** 2) ** 0.5
    let normalize_function = function(axis) {
        return function(d_x) {
            return (d_x - data_range[axis]['min']) / (data_range[axis]['max'] - data_range[axis]['min']) * chart_size[axis] / chart_diag;
        }
    }
    let normalized_data = {'x': data['date'].map(normalize_function('x')), 'y': data['value'].map(normalize_function('y'))};
    return normalized_data;
}
    