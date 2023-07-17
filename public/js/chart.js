"use strict"

var chartWidth;
var oldChartWidth;
var chartHeight;
const margin = {top: 170, bottom: 75, left: 90, right: 30};

const maxProminentFeatures = 10;

var valueField;

const PROMINENCE_FEATURE_COUNT = 5;

var prominenceThreshold = 0.12;
var prominentFeatures;

var chartSVG;

var prominentGuidesLayer;
var referenceGuidesLayer;
var prominentFeaturesLayer;
var referencedFeaturesLayer;
var plotHelpers;
var axes;
var chartLine;
var tooltipComponents;
var layout;

var miniChartG;
var miniPlotHelpers;
var miniAxes;
var miniXBrushArea;
var miniChartLine;

var miniValuesG;
var miniValuesScale;
var miniValuesAxes;
var miniYBrushArea;

var subData;
var dateExtent;
var miniXBrush;
var miniYBrush;

var layoutG;

function drawDefaultChart(chartData, metaData) {
    const chartIdSplit = chartData.split('|');
    const chartSet = chartIdSplit[0];
    const chartName = chartIdSplit[1];

    dataId = chartName;

    d3.csv("data/" + chartSet + "/data/" + chartName + ".csv", d => {
        const fields = Object.keys(d)
        valueField = fields[1 - fields.indexOf("Date")];

        d.date = genISODate(d.Date);
        d.value = +d[valueField];
        delete d.Date;
        delete d[valueField];

        return d;
    })
    .then(_data => {
        data = _data;
        zippedData = {'date': [], 'value': []}
        _data.forEach(_datum => {
            zippedData['date'].push(_datum['date'])
            zippedData['value'].push(_datum['value'])
        });
        subData = data;
        d3.json("data/" + chartSet + "/metadata/" + chartName + ".json")
        .then(function(_metadata) {
            metadata = _metadata;
            initChart();
        });
    });
}

async function initChart() {
    if (metadata === null) {
        metadata = {
            range: {
                x: {
                    min: data[0].date.toISOString().slice(0, 10),
                    max: data[data.length - 1].date.toISOString().slice(0, 10)
                },
                y: {
                    min: d3.min(data, d => d.value),
                    max: d3.max(data, d => d.value)
                }
            },
            size: {
                x: 500,
                y: 350
            }
        }
    }
    chartWidth = 500;
    oldChartWidth = 500;
    chartHeight = chartWidth * metadata.size.y / metadata.size.x;
    metadata.size.y = chartHeight;

    const xScale = d3.scaleTime()
        .domain([genISODate(metadata.range.x.min), genISODate(metadata.range.x.max)])
        .range([0, chartWidth]);
    const yScale = d3.scaleLinear()
        .domain([metadata.range.y.min, metadata.range.y.max])
        .range([chartHeight, 0]);

    const lineGenerator = d3.line()
        .x(d => xScale(d.date))
        .y(d => yScale(d.value));

    plotHelpers = {xScale: xScale, yScale: yScale, lineGenerator: lineGenerator}

    chartSVG = d3.select("#chart-container")
        .append('svg')
        .attr('width', chartWidth + margin.left + margin.right)
        .attr('height', chartHeight + margin.top + margin.bottom);

    const chartG = chartSVG
        .append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);

    layoutG = chartG.append('g')
        .attr('id', "layout");

    const layoutRect = layoutG.append('rect')
        .attr('x', -3)
        .attr('y', -190)
        .attr('width', chartWidth + 6)
        .attr('height', 175)
        .attr('fill', "#eee")
        .attr('stroke', "none");

    const layoutDivider = layoutG.append('path')
        .attr('d', `M${0},${-70}L${chartWidth},${-70}`)
        .attr('fill', "none")
        .attr('stroke-width', "1px")
        .attr('stroke', "gray");

    layout = {rect: layoutRect, divider: layoutDivider};

    layoutG.append('text')
        .attr('text-anchor', "middle")
        .attr('fill', "black")
        .attr('font-size', "12px")
        .attr('transform', `translate(-40, ${-46}) rotate(-90)`)
        .text("Visually");
    layoutG.append('text')
        .attr('text-anchor', "middle")
        .attr('fill', "black")
        .attr('font-size', "12px")
        .attr('transform', `translate(-30, ${-46}) rotate(-90)`)
        .text("Prominent")
    
    layoutG.append('text')
        .attr('text-anchor', "middle")
        .attr('fill', "black")
        .attr('font-size', "12px")
        .attr('transform', `translate(-40, ${-125}) rotate(-90)`)
        .text("Text")
    layoutG.append('text')
        .attr('text-anchor', "middle")
        .attr('fill', "black")
        .attr('font-size', "12px")
        .attr('transform', `translate(-30, ${-125}) rotate(-90)`)
        .text("References")

    for (let labelIdx = 0; labelIdx < 5; labelIdx++) {
        layoutG.append('text')
            .attr('id', `prominent-${labelIdx}-ranking-text`)
            .attr('class', "ranking-text highlight-default")
            .attr('text-anchor', "end")
            .attr('fill', "black")
            .attr('font-size', "9px")
            .attr('transform', `translate(-5, ${-22 - heightUnit * labelIdx})`)
            .text(`${labelIdx + 1}`);
    }

    const axesG = chartG.append('g')
        .attr('id', "axes-layer");
    const xAxis = axesG.append('g')
        .attr('id', "x-axis-layer")
        .attr('transform', `translate(0, ${chartHeight + 20})`)
        .call(d3.axisBottom(xScale));

    const xLabel = xAxis.append('text')
        .attr('text-anchor', "middle")
        .attr('fill', "black")
        .attr('font-size', "12px")
        .attr('transform', `translate(${chartWidth / 2}, 50)`)
        .text("Date")
    const yAxis = axesG.append('g')
        .attr('id', "y-axis-layer")
        .attr('transform', "translate(-20, 0)")
        .call(d3.axisLeft(yScale));
    const yLabel = yAxis.append('text')
        .attr('text-anchor', "middle")
        .attr('fill', "black")
        .attr('font-size', "12px")
        .attr('transform', `translate(-50, ${chartHeight / 2}) rotate(-90)`)
        .text(valueField);

    axes = {x: xAxis, y: yAxis, xLabel: xLabel, yLabel: yLabel};

    prominentGuidesLayer = chartG.append('g')
        .attr('id', 'prominentguides-layer');

    referenceGuidesLayer = chartG.append('g')
        .attr('id', 'referenceguides-layer');

    prominentFeaturesLayer = chartG.append('g')
        .attr('id', "prominentfeatures-layer");
    
    prominentFeatures = getProminentFeatures().filter(f => f.prominence > 1e-5);

    displayProminentFeatures(prominentFeatures, PROMINENCE_FEATURE_COUNT);

    referencedFeaturesLayer = chartG.append('g')
        .attr('id', "referencedfeatures-layer");

    chartLine = chartG.append('g')
        .attr('id', "chart-layer")
        .append('path')
        .datum(data)
        .attr('class', "data-line")
        .attr('d', d => lineGenerator(d));
    
    const tooltipG = chartG.append('g')
        .attr('id', "tooltip-group");
    
    const tooltipMark = tooltipG.append('circle')
        .attr('cx', -10000)
        .attr('cy', -10000)
        .attr('r', 5)
        .attr('fill', "black");
    const tooltipLine = tooltipG.append('path')
        .attr('d', `M${-10000},${0}L${-10000},${chartHeight + 20}`)
        .attr('stroke', "#444")
        .attr('stroke-width', "0.5");
    const tooltipTextYear = tooltipG.append('text')
        .attr('x', -10000)
        .attr('y', `${chartHeight + 35}`)
        .attr('class', "guide")
        .text("");
    const tooltipTextMD = tooltipG.append('text')
        .attr('x', -10000)
        .attr('y', `${chartHeight + 45}`)
        .attr('class', "guide")
        .text("");
    const tooltipToYAxis = tooltipG.append('path')
        .attr('d', `M${-10000},${-10000}L${-10000},${-10000}`)
        .attr('class', "guide line");
    const tooltipYAxisText = tooltipG.append('text')
        .attr('x', -30)
        .attr('y', -10000)
        .attr('class', "guide")
        .attr('text-anchor', "end")
        .text("");

    const tooltipArea = tooltipG.append('rect')
        .attr('x', -10)
        .attr('y', 0)
        .attr('width', chartWidth + 20)
        .attr('height', chartHeight + 20)
        .attr('opacity', 0)
        .on('mousemove', e => {
            const coords = d3.pointer(e);
            const mouseDate = plotHelpers.xScale.invert(coords[0]);
            var closestDatum = null;
            var closestDistance = Number.POSITIVE_INFINITY;
            for (let dataIdx = 0; dataIdx < data.length; dataIdx++) {
                let testDatum = data[dataIdx];
                let testDistance = Math.abs(testDatum.date - mouseDate);
                if (testDistance < closestDistance) {
                    closestDatum = testDatum;
                    closestDistance = testDistance;
                }
            }
            tooltipMark.attr('cx', plotHelpers.xScale(closestDatum.date))
                .attr('cy', plotHelpers.yScale(closestDatum.value));
            tooltipLine.attr('d', `M${plotHelpers.xScale(closestDatum.date)},${0}L${plotHelpers.xScale(closestDatum.date)},${chartHeight + 20}`);
            hideXTicks();
            hideYTicks();
            tooltipTextYear.attr('x', plotHelpers.xScale(closestDatum.date))
                .text(closestDatum.date.getUTCFullYear())
            tooltipTextMD.attr('x', plotHelpers.xScale(closestDatum.date))
                .text(`${`0${closestDatum.date.getUTCMonth() + 1}`.slice(-2)}/${`0${closestDatum.date.getUTCDate()}`.slice(-2)}`);
            tooltipToYAxis.attr('d', `M${-20},${plotHelpers.yScale(closestDatum.value)}L${plotHelpers.xScale(closestDatum.date)},${plotHelpers.yScale(closestDatum.value)}`);
            tooltipYAxisText.attr('y', plotHelpers.yScale(closestDatum.value))
                .text(closestDatum.value.toPrecision(4));
        })
        .on('mouseleave', e => {
            tooltipMark.attr('cx', -10000)
                .attr('cy', -10000);
            tooltipLine.attr('d', `M${-10000},${0}L${-10000},${chartHeight + 20}`);
            tooltipTextYear.attr('x', -10000);
            tooltipTextMD.attr('x', -10000);
            tooltipToYAxis.attr('d', `M${-10000},${-10000}L${-10000},${-10000}`);
            tooltipYAxisText.attr('y', -10000);
            showXTicks();
            showYTicks();
        });

    tooltipComponents = {area: tooltipArea, line: tooltipLine, yearText: tooltipTextYear, mdText: tooltipTextMD};

    initNoUiSliders();
}

function clearChart() {
    const chartContainer = document.getElementById("chart-container");
    while (chartContainer.lastElementChild) {
        chartContainer.removeChild(chartContainer.lastElementChild);
    }
}

function horizontalDimensionChange(newChartWidth) {
    if (newChartWidth === chartWidth) {
        return;
    }
    const dx = newChartWidth - chartWidth;
    oldChartWidth = chartWidth;
    chartWidth = newChartWidth;
    metadata.size.x = chartWidth;
    chartSVG.attr('width', chartWidth + margin.left + margin.right);

    plotHelpers.xScale = d3.scaleTime()
        .domain(dateExtent)
        .range([0, chartWidth]);

    plotHelpers.lineGenerator = d3.line()
        .defined(d => d.value >= metadata.range.y.min && d.value <= metadata.range.y.max)
        .x(d => plotHelpers.xScale(d.date))
        .y(d => plotHelpers.yScale(d.value));

    
    axes.x.call(d3.axisBottom(plotHelpers.xScale));
    axes.xLabel.attr('transform', `translate(${chartWidth / 2}, 50)`);
    
    chartLine.attr('d', d => plotHelpers.lineGenerator(d));

    tooltipComponents.area.attr('width', chartWidth + 20);


    layout.rect.attr('width', chartWidth + 6)
    layout.divider.attr('d', `M${0},${-70}L${chartWidth},${-70}`);


    prominentFeatures = getProminentFeatures().filter(f => f.prominence > 1e-5);
    clearProminentFeatures();
    displayProminentFeatures(prominentFeatures, PROMINENCE_FEATURE_COUNT); 


    if (references) {
        removeChartReferences();
        displayChartReferences();
    }
    
}

function verticalDimensionChange(newChartHeight) {
    if (newChartHeight === chartHeight) {
        return;
    }
    const dy = newChartHeight - chartHeight;
    chartHeight = newChartHeight;
    metadata.size.y = chartHeight;
    chartSVG.attr('height', chartHeight + margin.top + margin.bottom);

    plotHelpers.yScale = d3.scaleLinear()
        .domain([metadata.range.y.min, metadata.range.y.max])
        .range([chartHeight, 0]);

    plotHelpers.lineGenerator = d3.line()
        .defined(d => d.value >= metadata.range.y.min && d.value <= metadata.range.y.max)
        .x(d => plotHelpers.xScale(d.date))
        .y(d => plotHelpers.yScale(d.value));

    axes.x.attr('transform', `translate(0, ${chartHeight + 20})`);
    axes.y.call(d3.axisLeft(plotHelpers.yScale));
    axes.yLabel.attr('transform', `translate(-50, ${chartHeight / 2}) rotate(-90)`)
    
    chartLine.attr('d', d => plotHelpers.lineGenerator(d));

    tooltipComponents.area.attr('height', chartHeight + 20);
    tooltipComponents.line.attr('d', `M${-10000},${0}L${-10000},${chartHeight + 20}`);
    tooltipComponents.yearText.attr('y', `${chartHeight + 35}`);
    tooltipComponents.mdText.attr('y', `${chartHeight + 45}`);

    prominentFeatures = getProminentFeatures().filter(f => f.prominence > 1e-5);
    clearProminentFeatures();
    displayProminentFeatures(prominentFeatures, PROMINENCE_FEATURE_COUNT);

    if (references) {
        removeChartReferences();
        displayChartReferences();
    }

}

function clearProminentFeatures() {
    const prominentFeaturesLayerJS = document.getElementById("prominentfeatures-layer");
    while (prominentFeaturesLayerJS.lastElementChild) {
        prominentFeaturesLayerJS.removeChild(prominentFeaturesLayerJS.lastElementChild);
    }
    const prominentGuidesLayerJS = document.getElementById("prominentguides-layer");
    while (prominentGuidesLayerJS.lastElementChild) {
        prominentGuidesLayerJS.removeChild(prominentGuidesLayerJS.lastElementChild);
    }
}

function shiftPolyline(points, shift) {
    const slopes = [];
    for (let pointIdx = 0; pointIdx < points.length - 1; pointIdx++) {
        slopes.push((points[pointIdx + 1].y - points[pointIdx].y) / (points[pointIdx + 1].x - points[pointIdx].x));
    }
    const slopeSR = slopes.map(m => Math.sqrt(1 + m * m));
    const firstPoint = {x: points[0].x, y: points[0].y + shift * slopeSR[0]};
    const lastPoint = {x: points[points.length - 1].x, y: points[points.length - 1].y + shift * slopeSR[slopeSR.length - 1]};

    const shiftedPoints = [];
    shiftedPoints.push(firstPoint);

    var prevPoint = firstPoint;

    for (let slopeIdx = 0; slopeIdx < slopes.length - 1; slopeIdx++) {
        let m1 = slopes[slopeIdx];
        let m2 = slopes[slopeIdx + 1];
        if (Math.abs(m1 - m2) < 1e-5) {
            // If the slopes are essentially the same, no need to have a separate 'control point'
            continue;
        }
        let r1 = slopeSR[slopeIdx];
        let r2 = slopeSR[slopeIdx + 1];
        let currPoint = {x: points[slopeIdx + 1].x - shift * (r1 - r2) / (m1 - m2), y: points[slopeIdx + 1].y + shift * (m1 * r2 - m2* r1) / (m1 - m2)};
        if (currPoint.x > prevPoint.x && currPoint.x < lastPoint.x) {
            // Make sure we remove 'shoot-backs' from features being smaller than the shift amount
            shiftedPoints.push(currPoint);
            prevPoint = currPoint;
        }
    }
    shiftedPoints.push(lastPoint);

    return shiftedPoints;
}

function displayTrend(layer, feature, id, styleClass, shiftUp, mouseEvents = null) {
    const startDate = genISODate(feature[0]);
    const endDate = genISODate(feature[1]);
    const dataInRange = data.filter(d => d.date >= startDate && d.date <= endDate);

    const mappedPoints = dataInRange.map(d => ({x: plotHelpers.xScale(d.date), y: plotHelpers.yScale(d.value)}));
    const shiftedPointsFar = shiftPolyline(mappedPoints, -0.5 * Math.sign(shiftUp) - shiftUp * 4);
    const shiftedPointsClose = shiftPolyline(mappedPoints, -0.5 * Math.sign(shiftUp));
    
    const trendG = layer.append('g')
        .attr('id', `${id}-group`);
    addTrendGuide(trendG, feature, mappedPoints[0], mappedPoints[mappedPoints.length - 1], id);

    const trend = trendG.append('path')
        .datum(shiftedPointsFar.concat(shiftedPointsClose.reverse()))
        .attr('d', d => d3.line()
            .x(d => d.x)
            .y(d => d.y)(d))
        .attr('id', `${id}-chart`)
        .attr('class', styleClass);

    if (mouseEvents) {
        for (let eventName in mouseEvents) {
            trend.on(eventName, mouseEvents[eventName]);
        }
    }
    
    displayTrendOnAxis(trendG, feature, id, styleClass, shiftUp, mouseEvents);
}

function displayTrendOnAxis(layer, feature, id, styleClass, shiftUp, mouseEvents = null) {
    const startDate = genISODate(feature[0]);
    const endDate = genISODate(feature[1]);
    
    const shiftedPointsFar = [{x: plotHelpers.xScale(startDate), y: chartHeight + 20 - 0.5 * Math.sign(shiftUp) - shiftUp * 3}, {x: plotHelpers.xScale(endDate), y: chartHeight + 20 - 0.5 * Math.sign(shiftUp) - shiftUp * 3}];
    const shiftedPointsClose = [{x: plotHelpers.xScale(startDate), y: chartHeight + 20 - 0.5 * Math.sign(shiftUp)}, {x: plotHelpers.xScale(endDate), y: chartHeight + 20 - 0.5 * Math.sign(shiftUp)}];

    const trendOnAxis = layer.append('path')
        .datum(shiftedPointsFar.concat(shiftedPointsClose.reverse()))
        .attr('d', d => d3.line()
            .x(d => d.x)
            .y(d => d.y)(d))
        .attr('id', `${id}-chart-axis`)
        .attr('class', styleClass);
    
    if (mouseEvents) {
        for (let eventName in mouseEvents) {
            trendOnAxis.on(eventName, mouseEvents[eventName]);
        }
    }
}

function displayPoint(layer, feature, id, styleClass, topArc, mouseEvents = null) {
    const featureDate = genISODate(feature);
    const featureData = data.filter(d => d.date.getTime() == featureDate.getTime())[0];

    var arc = d3.arc()
        .innerRadius(0)
        .outerRadius(11)
        .startAngle(0.5 * Math.PI + (topArc ? 0 : Math.PI))
        .endAngle(1.5 * Math.PI + (topArc ? 0 : Math.PI));
    const pointG = layer.append('g')
        .attr('id', `${id}-group`);

    addPointGuide(pointG, feature, {x: plotHelpers.xScale(featureData.date), y: plotHelpers.yScale(featureData.value)}, id)

    const point = pointG.append('path')
        .attr('transform', `translate(${plotHelpers.xScale(featureData.date)}, ${plotHelpers.yScale(featureData.value)})`)
        .attr('d', arc)
        .attr('id', `${id}-chart`)
        .attr('class', styleClass);
        
    if (mouseEvents) {
        for (let eventName in mouseEvents) {
            point.on(eventName, mouseEvents[eventName]);
        }
    }
    displayPointOnAxis(pointG, feature, id, styleClass, topArc, mouseEvents);
}

function displayPointOnAxis(layer, feature, id, styleClass, topArc, mouseEvents = null) {
    const featureDate = genISODate(feature);

    var arc = d3.arc()
        .innerRadius(0)
        .outerRadius(6)
        .startAngle(0.5 * Math.PI + (topArc ? 0 : Math.PI))
        .endAngle(1.5 * Math.PI + (topArc ? 0 : Math.PI));
    const pointOnAxis = layer.append('path')
        .attr('transform', `translate(${plotHelpers.xScale(featureDate)}, ${chartHeight + 20})`)
        .attr('d', arc)
        .attr('id', `${id}-chart-axis`)
        .attr('class', styleClass);
    
    if (mouseEvents) {
        for (let eventName in mouseEvents) {
            pointOnAxis.on(eventName, mouseEvents[eventName]);
        }
    }
}

function generateHighlightReferenceMouseenterEvent(matchingFeatureIdx = null) {
    function highlightReferenceMouseenterEvent() {
        const idSplit = this.id.split("-");
        const sentenceIdx = idSplit[1];
        const referenceIdx = idSplit[2];
        modifyHighlightingProminentFeatures("highlight-deemph");
        modifyHighlightingReferences("highlight-deemph");
        modifyHighlightingMark("highlight-deemph");
        modifyHighlightingReferences("highlight-emph", sentenceIdx, referenceIdx);
        modifyHighlightingMark("highlight-emph", sentenceIdx, referenceIdx);
        if (matchingFeatureIdx !== null) {
            modifyHighlightingProminentFeatures("highlight-emph", matchingFeatureIdx);
        }
        hideXTicks();
    }
    return highlightReferenceMouseenterEvent;
}

function generateHighlightReferenceMouseleaveEvent(matchingFeatureIdx = null) {
    function highlightReferenceMouseleaveEvent() {
        const idSplit = this.id.split("-");
        const sentenceIdx = idSplit[1];
        const referenceIdx = idSplit[2];
        modifyHighlightingReferences("highlight-default");
        modifyHighlightingProminentFeatures("highlight-default");
        modifyHighlightingMark("highlight-default");
        showXTicks();
    }
    return highlightReferenceMouseleaveEvent;
}


function generateHighlightProminentFeatureMouseenterEvent(matchingReferenceIdx = null) {
    function highlightProminentFeatureMouseenterEvent() {
        const idSplit = this.id.split("-");
        const featureIdx = idSplit[1];
        modifyHighlightingProminentFeatures("highlight-deemph");
        modifyHighlightingReferences("highlight-deemph");
        modifyHighlightingMark("highlight-deemph");
        modifyHighlightingProminentFeatures("highlight-emph", featureIdx);
        if (matchingReferenceIdx !== null) {
            const idxSplit = matchingReferenceIdx.split("-");
            modifyHighlightingReferences("highlight-emph", idxSplit[0], idxSplit[1]);
            modifyHighlightingMark("highlight-emph", idxSplit[0], idxSplit[1]);
        }
        hideXTicks();
    }
    return highlightProminentFeatureMouseenterEvent;
}

function generateHighlightProminentFeatureMouseleaveEvent(matchingReferenceIdx = null) {
    function highlightProminentFeatureMouseleaveEvent() {
        const idSplit = this.id.split("-");
        const sentenceIdx = idSplit[1];
        const referenceIdx = idSplit[2];
        modifyHighlightingReferences("highlight-default");
        modifyHighlightingProminentFeatures("highlight-default");
        modifyHighlightingMark("highlight-default");
        showXTicks();
    }
    return highlightProminentFeatureMouseleaveEvent;
}


function modifyHighlightingProminentFeatures(newHighlight, featureIdx = null) {
    var identifier;
    if (featureIdx === null) {
        identifier = "'prominent-'";
    } else {
        identifier = `'prominent-${featureIdx}-'`;
    }

    const elems = document.querySelectorAll(`*[id^=${identifier}]`);
    for (let elemIdx = 0; elemIdx < elems.length; elemIdx++) {
        let elem = elems[elemIdx];
        elem.classList.remove("highlight-default", "highlight-emph", "highlight-deemph", "highlight-outfocus");
        elem.classList.add(newHighlight);
    }
}

function modifyHighlightingReferences(newHighlight, sentenceIdx = null, referenceIdx = null) {
    var identifier;
    if (sentenceIdx === null) {
        identifier = "'reference-'";
    } else if (referenceIdx === null) {
        identifier = `'reference-${sentenceIdx}-'`;
    } else {
        identifier = `'reference-${sentenceIdx}-${referenceIdx}'`
    }

    const elems = document.querySelectorAll(`*[id^=${identifier}]`);
    for (let elemIdx = 0; elemIdx < elems.length; elemIdx++) {
        let elem = elems[elemIdx];
        elem.classList.remove("highlight-default", "highlight-emph", "highlight-deemph", "highlight-outfocus");
        elem.classList.add(newHighlight);
    }
}

function removeChartReferences(sentenceIdx = null, referenceIdx = null) {
    var identifier;
    if (sentenceIdx === null) {
        identifier = "'reference-'";
    } else if (referenceIdx === null) {
        identifier = `'reference-${sentenceIdx}-'`;
    } else {
        identifier = `'reference-${sentenceIdx}-${referenceIdx}'`
    }

    const elems = document.querySelectorAll(`*[id^=${identifier}]`);
    for (let elemIdx = 0; elemIdx < elems.length; elemIdx++) {
        let elem = elems[elemIdx];
        elem.remove();
    }
}

function hideXTicks() {
    const xTicks = document.getElementById("x-axis-layer").querySelectorAll(`g.tick`);
    for (let tickIdx = 0; tickIdx < xTicks.length; tickIdx++) {
        let xTick = xTicks[tickIdx];
        xTick.classList.add("hidden");
    }
}

function showXTicks() {
    const xTicks = document.getElementById("x-axis-layer").querySelectorAll(`g.tick`);
    for (let tickIdx = 0; tickIdx < xTicks.length; tickIdx++) {
        let xTick = xTicks[tickIdx];
        xTick.classList.remove("hidden");
    }
}

function hideYTicks() {
    const yTicks = document.getElementById("y-axis-layer").querySelectorAll(`g.tick`);
    for (let tickIdx = 0; tickIdx < yTicks.length; tickIdx++) {
        let yTick = yTicks[tickIdx];
        yTick.classList.add("hidden");
    }
}

function showYTicks() {
    const yTicks = document.getElementById("y-axis-layer").querySelectorAll(`g.tick`);
    for (let tickIdx = 0; tickIdx < yTicks.length; tickIdx++) {
        let yTick = yTicks[tickIdx];
        yTick.classList.remove("hidden");
    }
}

function hideProminentFeatures() {
    prominentFeaturesLayer.style('visibility', "hidden");
    prominentGuidesLayer.style('visibility', "hidden");
}

function showProminentFeatures() {
    prominentFeaturesLayer.style('visibility', "visible");
    prominentGuidesLayer.style('visibility', "visible");
}

function hideReferences() {
    referencedFeaturesLayer.style('visibility', "hidden");
    referenceGuidesLayer.style('visibility', "hidden");
}

function showReferences() {
    referencedFeaturesLayer.style('visibility', "visible");
    referenceGuidesLayer.style('visibility', "visible");
}

var prominentTrends = null;
var prominentPoints = null;
function displayProminentFeatures(prominentFeatures, prominenceThreshold) {
    var trends = [];
    var points = [];
    for (let prominentFeatureIdx = 0; prominentFeatureIdx < Math.min(prominenceThreshold, prominentFeatures.length); prominentFeatureIdx++) {
        let prominentFeature = prominentFeatures[prominentFeatureIdx];
        let prominentDegree;
        if (prominentFeature.prominence > 0.12) {
            prominentDegree = "high";
        } else if (prominentFeature.prominence > 0.09) {
            prominentDegree = "med";
        } else {
            prominentDegree = "low";
        }
        if (Array.isArray(prominentFeature.dates)) {
            trends.push({
                range: [genISODate(prominentFeature.dates[0]), genISODate(prominentFeature.dates[1])],
                type: prominentFeature.type,
                height: prominentFeatureIdx + 1,
                prominentIdx: prominentFeatureIdx,
                id: `prominent-${prominentFeatureIdx}`,
                styleClass: `trend prominent-${prominentDegree} prominent-${prominentFeatureIdx} highlight-default`,
                mouseEvents: {'mouseenter': generateHighlightProminentFeatureMouseenterEvent, 'mouseleave': generateHighlightProminentFeatureMouseleaveEvent},
                prominence: prominentFeature.prominence,
                pair: null
            });
        } else {
            points.push({
                location: genISODate(prominentFeature.dates),
                type: prominentFeature.type,
                height: prominentFeatureIdx + 1,
                prominentIdx: prominentFeatureIdx,
                id: `prominent-${prominentFeatureIdx}`,
                styleClass: `point prominent-${prominentDegree} prominent-${prominentFeatureIdx} highlight-default`,
                mouseEvents: {'mouseenter': generateHighlightProminentFeatureMouseenterEvent, 'mouseleave': generateHighlightProminentFeatureMouseleaveEvent},
                prominence: prominentFeature.prominence,
                pair: null
            });
        }
    }
    prominentTrends = trends;
    prominentPoints = points;
    displayRaisedTrends(prominentTrends, false);
    displayRaisedPoints(prominentPoints, false);
}

function displayChartReferences() {
    var trends = [];
    var points = [];
    references.forEach(chartReferences => {
        chartReferences.references.forEach(reference => {
            if (Array.isArray(reference.chart)) {
                trends.push({
                    range: [genISODate(reference.chart[0]), genISODate(reference.chart[1])],
                    type: reference.featureType,
                    dependent: false,
                    referenceIdx: reference.referenceIdx,
                    id: `reference-${reference.referenceIdx}`,
                    styleClass: `trend reference reference-${reference.referenceIdx} highlight-default coloring-${reference.numbering % 8}`,
                    mouseEvents: {'mouseenter': generateHighlightReferenceMouseenterEvent, 'mouseleave': generateHighlightReferenceMouseleaveEvent},
                    pair: null
                });
                if (reference.mentions[0] === true) {
                    points.push({
                        location: genISODate(reference.chart[0]),
                        type: "inflection:",
                        dependent: true,
                        referenceIdx: reference.referenceIdx,
                        id: `reference-${reference.referenceIdx}-start`,
                        styleClass: `point reference reference-${reference.referenceIdx} highlight-default coloring-${reference.numbering % 8}`,
                        mouseEvents: {'mouseenter': generateHighlightReferenceMouseenterEvent, 'mouseleave': generateHighlightReferenceMouseleaveEvent},
                        pair: null
                    });
                }
                if (reference.mentions[1] === true) {
                    points.push({
                        location: genISODate(reference.chart[1]),
                        type: "inflection:",
                        dependent: true,
                        referenceIdx: reference.referenceIdx,
                        id: `reference-${reference.referenceIdx}-end`,
                        styleClass: `point reference reference-${reference.referenceIdx} highlight-default coloring-${reference.numbering % 8}`,
                        mouseEvents: {'mouseenter': generateHighlightReferenceMouseenterEvent, 'mouseleave': generateHighlightReferenceMouseleaveEvent},
                        pair: null
                    });
                }
            } else {
                points.push({
                    location: genISODate(reference.chart),
                    type: reference.featureType,
                    dependent: false,
                    referenceIdx: reference.referenceIdx,
                    id: `reference-${reference.referenceIdx}`,
                    styleClass: `point reference reference-${reference.referenceIdx} highlight-default coloring-${reference.numbering % 8}`,
                    mouseEvents: {'mouseenter': generateHighlightReferenceMouseenterEvent, 'mouseleave': generateHighlightReferenceMouseleaveEvent},
                    pair: null
                });
            }
        })
    });
    const displayHeights = computeReferenceDisplayHeights(trends, points);
    trends = displayHeights[0];
    points = displayHeights[1];

    displayRaisedTrends(trends, true);
    displayRaisedPoints(points, true);

    const matchedTrendIds = trends.filter(t => t.matched).map(t => t.id);
    const matchedPointIds = points.filter(p => p.matched).map(p => p.id);
    matchedTrendIds.concat(matchedPointIds).forEach(matchedFeatureId => {
        const targetSpans = document.getElementById("text-container").getElementsByClassName(matchedFeatureId);
        for (let targetSpanIdx = 0; targetSpanIdx < targetSpans.length; targetSpanIdx++) {
            targetSpans[targetSpanIdx].classList.remove("unmatched");
        }
    });

    const unmatchedTrendIds = trends.filter(t => !t.matched).map(t => t.id);
    const unmatchedPointIds = points.filter(p => !(p.dependent || p.matched)).map(p => p.id);
    unmatchedTrendIds.concat(unmatchedPointIds).forEach(unmatchedFeatureId => {
        const targetSpans = document.getElementById("text-container").getElementsByClassName(unmatchedFeatureId);
        for (let targetSpanIdx = 0; targetSpanIdx < targetSpans.length; targetSpanIdx++) {
            targetSpans[targetSpanIdx].classList.add("unmatched");
        }
    });
}

const heightUnit = 8;
const dispWidth = 8;
function displayRaisedTrends(trends, shifted) {
    const layer = (shifted ? referencedFeaturesLayer : prominentFeaturesLayer);
    const guidesLayer = (shifted ? referenceGuidesLayer : prominentGuidesLayer);
    const shift = 0;
    trends.reverse().forEach(trend => {
        const quadCorners = [
            {x: plotHelpers.xScale(trend.range[0]), y: -(trend.height - 1) * heightUnit - dispWidth - shift + 1 - 22}, 
            {x: plotHelpers.xScale(trend.range[1]), y: -(trend.height - 1) * heightUnit - dispWidth - shift + 1 - 22},
            {x: plotHelpers.xScale(trend.range[1]), y: -(trend.height - 1) * heightUnit - shift - 22},
            {x: plotHelpers.xScale(trend.range[0]), y: -(trend.height - 1) * heightUnit - shift - 22},
            {x: plotHelpers.xScale(trend.range[0]), y: -(trend.height - 1) * heightUnit - dispWidth - shift + 1 - 22}
        ];
        const raisedTrend = layer.append('path')
            .datum(quadCorners)
            .attr('d', d => d3.line()
                .x(d => d.x)
                .y(d => d.y)(d))
            .attr('id', `${trend.id}-chart-axis`)
            .attr('class', trend.styleClass);
        for (let eventName in trend.mouseEvents) {
            raisedTrend.on(eventName, trend.mouseEvents[eventName]((trend.pair === null ? null: trend.pair.prominentIdx)));
        }
        addTrendOnLine(layer, trend, shifted);
        addTrendGuide(guidesLayer, trend);
    });
}

function displayRaisedPoints(points, shifted) {
    const layer = (shifted ? referencedFeaturesLayer : prominentFeaturesLayer);
    const guidesLayer = (shifted ? referenceGuidesLayer : prominentGuidesLayer);
    const shift = 0;
    points.forEach(point => {
        const raisedPoint = layer.append('circle')
            .attr('cx', plotHelpers.xScale(point.location))
            .attr('cy', -((point.height - 1) * heightUnit) - shift - dispWidth / 2 - 22)
            .attr('r', dispWidth / 2)
            .attr('id', `${point.id}-chart-axis`)
            .attr('class', point.styleClass);
        for (let eventName in point.mouseEvents) {
            raisedPoint.on(eventName, point.mouseEvents[eventName]((point.pair === null ? null: point.pair.prominentIdx)));
        }
        addPointOnLine(layer, point, shifted);
        addPointGuide(guidesLayer, point);
    });
}

function addPointOnLine(layer, point, shifted) {
    point.value = data.filter(d => d.date.toUTCString() === point.location.toUTCString())[0].value;
    const pointOnLine = layer.append('circle')
        .attr('cx', plotHelpers.xScale(point.location))
        .attr('cy', plotHelpers.yScale(point.value))
        .attr('r', (shifted ? 4 : 8))
        .attr('id', `${point.id}-chart`)
        .attr('class', `on-line ${point.styleClass}`);
    for (let eventName in point.mouseEvents) {
        pointOnLine.on(eventName, point.mouseEvents[eventName]((point.pair === null ? null: point.pair.prominentIdx)));
    }
}


function addPointGuide(layer, point) {
    if (point.matched) {
        layer.append('path')
            .attr('d', `M${plotHelpers.xScale(point.location)},${chartHeight + 20}L${plotHelpers.xScale(point.location)},${-(point.pair.height - 1) * heightUnit - 22}`)
            .attr('id', `${point.id}-guide-point-line`)
            .attr('class', "guide line highlight-default");
        layer.append('path')
            .attr('d', `M${plotHelpers.xScale(point.location)},${-(point.pair.height - 1) * heightUnit - 22}L${plotHelpers.xScale(point.location)},${-(point.height - 1) * heightUnit - 22}`)
            .attr('id', `${point.id}-guide-point-line-match`)
            .attr('class', "guide line highlight-default")
            .style('stroke-width', 2)
            .style('stroke', "black");
    } else {
        layer.append('path')
            .attr('d', `M${plotHelpers.xScale(point.location)},${chartHeight + 20}L${plotHelpers.xScale(point.location)},${-(point.height - 1) * heightUnit - 22}`)
            .attr('id', `${point.id}-guide-point-line`)
            .attr('class', "guide line highlight-default");
    }
    layer.append('text')
        .attr('x', `${plotHelpers.xScale(point.location)}`)
        .attr('y', `${chartHeight + 35}`)
        .attr('id', `${point.id}-guide-point-text-year`)
        .attr('class', "guide point highlight-default")
        .text(point.location.getUTCFullYear());
    layer.append('text')
        .attr('x', `${plotHelpers.xScale(point.location)}`)
        .attr('y', `${chartHeight + 45}`)
        .attr('id', `${point.id}-guide-point-text-mthday`)
        .attr('class', "guide point highlight-default")
        .text(`${`0${point.location.getUTCMonth() + 1}`.slice(-2)}/${`0${point.location.getUTCDate()}`.slice(-2)}`);
}

function addTrendOnLine(layer, trend, shifted) {
    const trendData = data.filter(d => (d.date >= trend.range[0] && d.date <= trend.range[1]));
    trend.values = [trendData[0].value, trendData[trendData.length - 1].value];
    const trendOnLine = layer.append('path')
        .datum(trendData)
        .attr('id', `${trend.id}-chart`)
        .attr('class', `on-line ${(shifted ? "reference" : "prominent")} ${trend.styleClass}`)
        .attr('d', d => plotHelpers.lineGenerator(d));
    for (let eventName in trend.mouseEvents) {
        trendOnLine.on(eventName, trend.mouseEvents[eventName]((trend.pair === null ? null: trend.pair.prominentIdx)));
    }

}

function addTrendGuide(layer, trend, shifted) {
    if (trend.matched) {
        layer.append('path')
            .attr('d', `M${plotHelpers.xScale(trend.range[0])},${chartHeight + 20}L${plotHelpers.xScale(trend.range[0])},${-(trend.pair.height - 1) * heightUnit - 22}`)
            .attr('id', `${trend.id}-guide-start-line`)
            .attr('class', "guide line highlight-default");
    
        layer.append('path')
            .attr('d', `M${plotHelpers.xScale(trend.range[1])},${chartHeight + 20}L${plotHelpers.xScale(trend.range[1])},${-(trend.pair.height - 1) * heightUnit - 22}`)
            .attr('id', `${trend.id}-guide-start-line`)
            .attr('class', "guide line highlight-default");

        layer.append('path')
            .attr('d', `M${plotHelpers.xScale(trend.range[0])},${-(trend.pair.height) * heightUnit - 22}L${plotHelpers.xScale(trend.range[0])},${-(trend.height - 1) * heightUnit - 22}`)
            .attr('id', `${trend.id}-guide-start-line-match`)
            .attr('class', "guide line highlight-default")
            .style('stroke-width', 2)
            .style('stroke', "black")

        layer.append('path')
            .attr('d', `M${plotHelpers.xScale(trend.range[1])},${-(trend.pair.height) * heightUnit - 22}L${plotHelpers.xScale(trend.range[1])},${-(trend.height - 1) * heightUnit - 22}`)
            .attr('id', `${trend.id}-guide-start-line-match`)
            .attr('class', "guide line highlight-default")
            .style('stroke-width', 2)
            .style('stroke', "black")
    }
    else {
        layer.append('path')
            .attr('d', `M${plotHelpers.xScale(trend.range[0])},${chartHeight + 20}L${plotHelpers.xScale(trend.range[0])},${-(trend.height - 1) * heightUnit - 22}`)
            .attr('id', `${trend.id}-guide-start-line`)
            .attr('class', "guide line highlight-default");
    
        layer.append('path')
            .attr('d', `M${plotHelpers.xScale(trend.range[1])},${chartHeight + 20}L${plotHelpers.xScale(trend.range[1])},${-(trend.height - 1) * heightUnit - 22}`)
            .attr('id', `${trend.id}-guide-start-line`)
            .attr('class', "guide line highlight-default");
    }

    layer.append('text')
        .attr('x', `${plotHelpers.xScale(trend.range[0])}`)
        .attr('y', `${chartHeight + 35}`)
        .attr('id', `${trend.id}-guide-start-text-year`)
        .attr('class', "guide start highlight-default")
        .text(trend.range[0].getUTCFullYear());
    layer.append('text')
        .attr('x', `${plotHelpers.xScale(trend.range[0])}`)
        .attr('y', `${chartHeight + 45}`)
        .attr('id', `${trend.id}-guide-start-text-mthday`)
        .attr('class', "guide start highlight-default")
        .text(`${`0${trend.range[0].getUTCMonth() + 1}`.slice(-2)}/${`0${trend.range[0].getUTCDate()}`.slice(-2)}`);

    layer.append('text')
        .attr('x', `${plotHelpers.xScale(trend.range[1])}`)
        .attr('y', `${chartHeight + 35}`)
        .attr('id', `${trend.id}-guide-end-text-year`)
        .attr('class', "guide end highlight-default")
        .text(trend.range[1].getUTCFullYear());
    layer.append('text')
        .attr('x', `${plotHelpers.xScale(trend.range[1])}`)
        .attr('y', `${chartHeight + 45}`)
        .attr('id', `${trend.id}-guide-end-text-mthday`)
        .attr('class', "guide end highlight-default")
        .text(`${`0${trend.range[1].getUTCMonth() + 1}`.slice(-2)}/${`0${trend.range[1].getUTCDate()}`.slice(-2)}`);
}

function computeReferenceDisplayHeights(trends, points) {
    let nextTrendHeight = 8;
    let features = [].concat(trends, points);
    features.sort((a, b) => {
        if (a.id > b.id) {
            return 1;
        }
        if (a.id < b.id) {
            return -1;
        }
        if ("location" in a && !("location" in b)) {
            return -1;
        }
        if (!("location" in a) && "location" in b) {
            return 1;
        }
        else {
            return 0;
        }
    });
    var lastDependent = false;
    features.forEach(f => {
        if (!f.dependent) {
            f.height = nextTrendHeight;
            nextTrendHeight += 1;
            lastDependent = false;
        } else {
            if (lastDependent) {
                nextTrendHeight -= 1;
            }
            f.height = nextTrendHeight;
            nextTrendHeight += 1;
            lastDependent = true;
        }

    })

    trends.forEach(trend => {
        trend.matched = false;
        for (let prominentTrendIdx = 0; prominentTrendIdx < prominentTrends.length; prominentTrendIdx++) {
            let prominentTrend = prominentTrends[prominentTrendIdx];
                if (trend.range[0] >= prominentTrend.range[1] || trend.range[1] <= prominentTrend.range[0]) {
                    continue;
                }
                let intersection = [Math.max(trend.range[0], prominentTrend.range[0]), Math.min(trend.range[1], prominentTrend.range[1])];
                let union = [Math.min(trend.range[0], prominentTrend.range[0]), Math.max(trend.range[1], prominentTrend.range[1])];
                let iou = (intersection[1] - intersection[0]) / (union[1] - union[0]);
                if (iou > 0.95) {
                    trend.matched = true;
                    trend.pair = prominentTrend;
                    const elems = document.querySelectorAll(`*[id^='${prominentTrend.id}-']`);
                    for (let elemIdx = 0; elemIdx < elems.length; elemIdx++) {
                        elems[elemIdx].classList.add("matched");
                        elems[elemIdx].onmouseenter = generateHighlightProminentFeatureMouseenterEvent(trend.referenceIdx);
                        elems[elemIdx].onmouseleave = generateHighlightProminentFeatureMouseleaveEvent(trend.referenceIdx);
                    }
                    break;
                }
        }
    });
    points.forEach(point => {
        point.matched = false;
        for (let prominentPointIdx = 0; prominentPointIdx < prominentPoints.length; prominentPointIdx++) {
            let prominentPoint = prominentPoints[prominentPointIdx];
            if (point.location.toUTCString() === prominentPoint.location.toUTCString()) {
                point.matched = true;
                point.pair = prominentPoint;
                const elems = document.querySelectorAll(`*[id^='${prominentPoint.id}-']`);
                for (let elemIdx = 0; elemIdx < elems.length; elemIdx++) {
                    elems[elemIdx].classList.add("matched");
                    elems[elemIdx].onmouseenter = generateHighlightProminentFeatureMouseenterEvent(point.referenceIdx);
                    elems[elemIdx].onmouseleave = generateHighlightProminentFeatureMouseleaveEvent(point.referenceIdx);
                }
                break;
            }
        }
    });
    return [trends, points];
}