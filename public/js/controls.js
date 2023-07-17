"use strict"

var fileReader;
const localeStringOptions = { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
var dataId;
var data;
var zippedData;
var metadata;

function initDropZone() {
    fileReader = new FileReader();
    fileReader.onload = (event) => {
        const contents = event.target.result;
        processData(contents);
    };

    // Get the drop zone element
    var dropZone = document.getElementById('drop-zone');

    // Add a listener for the dragover event
    dropZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    // Add a listener for the dragleave event
    dropZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    });

    // Add a listener for the drop event
    dropZone.addEventListener('drop', function(e) {
        e.preventDefault();

        // Check that only one file was dropped
        if (e.dataTransfer.files.length !== 1) {
            alert('Please drop only one file.');
            return;
        }

        // Check that the file type is CSV
        if (e.dataTransfer.files[0].type !== 'text/csv') {
            alert('Please drop only CSV files.');
            return;
        }

        // Get the dropped file
        var file = e.dataTransfer.files[0];

        // TODO: Handle the dropped file
        dataId = file.name.slice(0, -3);
        fileReader.readAsText(file);

        // Remove the drag-over class from the drop zone
        dropZone.classList.remove('drag-over');
    });

}

function initNoUiSliders() {
    const chartWidthSlider = document.getElementById("chart-width-slider-container");
    const chartHeightSlider = document.getElementById("chart-height-slider-container");
    const timeRangeSlider = document.getElementById("time-range-slider-container");
    const valueRangeSlider = document.getElementById("value-range-slider-container");

    if (chartWidthSlider.noUiSlider) {
        chartWidthSlider.noUiSlider.set(chartWidth);
        chartHeightSlider.noUiSlider.set(chartHeight);

        const minDate = genISODate(metadata.range.x.min).getTime()
        const maxDate = genISODate(metadata.range.x.max).getTime()
        timeRangeSlider.noUiSlider.updateOptions({
            range: {
                min: minDate,
                max: maxDate
            },
            step: 1000 * 60 * 60 * 24,
            start: [minDate, maxDate]
        });

        const dataRange = metadata.range.y.max - metadata.range.y.min;
        const stepSize = 10 ** (Math.floor(Math.log10(dataRange)) - 1);
        const valueExtent = [dataRange / metadata.range.y.max < 0.33 ? metadata.range.y.min - 0.5 * dataRange : Math.min(metadata.range.y.min - 0.5 * dataRange, 0), dataRange < 33 ? metadata.range.y.max + 0.5 * dataRange : Math.max(metadata.range.y.max + 0.5 * dataRange, 100)];
        valueExtent[0] = Math.floor(valueExtent[0] / stepSize) * stepSize;
        valueExtent[1] = Math.ceil(valueExtent[1] / stepSize) * stepSize;
        const tempMax = Math.ceil(metadata.range.y.max / stepSize) * stepSize;
        const tempMin = Math.floor(metadata.range.y.min / stepSize) * stepSize;
        
        valueRangeSlider.noUiSlider.updateOptions({
            range: {
                min: -valueExtent[1],
                max: -valueExtent[0]
            },
            step: stepSize,
            margin: stepSize,
            format: {
                to: function (value) {
                    return - value.toFixed(stepSize > 1 ? 0 : -Math.round(Math.log10(stepSize)));
                },
                from: function (value) {
                    return -value;
                }
            },
        });

        valueRangeSlider.noUiSlider.set([tempMax, tempMin]);
    } else {
        noUiSlider.create(chartWidthSlider, {
            range: {
                min: 0,
                max: 600
            },
            step: 10,
            start: [chartWidth],
            format: {
                to: function (value) {
                    return Math.round(value/ 10) * 10;
                },
                from: function (value) {
                    return +value;
                }
            },
            connect: true,
            orientation: 'horizontal',
            behavior: 'drag',
            tooltips: true
        });
    
        chartWidthSlider.noUiSlider.on('set', (values, handle) => {
            if (values[handle] < 70) {
                chartWidthSlider.noUiSlider.set(70);
            }        
        });
    
        chartWidthSlider.noUiSlider.on('update', (values, handle) => {
            if (values[handle] < 70) {
                return;
            }
           
            horizontalDimensionChange(values[handle]);
        });
    
    
        noUiSlider.create(chartHeightSlider, {
            range: {
                min: 0,
                max: 600
            },
            step: 10,
            start: [chartHeight],
            format: {
                to: function (value) {
                    return Math.round(value/ 10) * 10;
                },
                from: function (value) {
                    return +value;
                }
            },
            connect: true,
            orientation: 'vertical',
            behavior: 'drag',
            tooltips: true
        });
    
        chartHeightSlider.noUiSlider.on('set', (values, handle) => {
            if (values[handle] < 70) {
                chartHeightSlider.noUiSlider.set(70);
            }        
        });
    
        chartHeightSlider.noUiSlider.on('update', (values, handle) => {
            if (values[handle] < 70) {
                return;
            }
           
            verticalDimensionChange(values[handle]);
        });
    
        const minDate = genISODate(metadata.range.x.min).getTime();
        const maxDate = genISODate(metadata.range.x.max).getTime();
        noUiSlider.create(timeRangeSlider, {
            range: {
                min: minDate,
                max: maxDate
            },
            step: 1000 * 60 * 60 * 24,
            start: [minDate, maxDate],
            format: {
                to: function (value) {
                    return genISODate(value).toISOString().slice(0, 10);
                },
                from: function (value) {
                    return genISODate(+value).getTime();
                }
            },
            connect: true,
            orientation: 'horizontal',
            behavior: 'drag',
            tooltips: true
        });
    
        timeRangeSlider.noUiSlider.on('update', (values, handle) => {
            const newMinDate = genISODate(values[0]);
            const newMaxDate = genISODate(values[1]);
            metadata.range.x.min = newMinDate;
            metadata.range.x.max = newMaxDate;
            dateExtent = [newMinDate, newMaxDate];
    
            const testSubData = data.filter(d => d.date >= newMinDate && d.date <= newMaxDate);
            if (testSubData.length < 2) {
                document.getElementById("time-range-empty-warning").style.display = "block";
            } else {
                document.getElementById("time-range-empty-warning").style.display = "none";
            }
            subData = testSubData;

            document.getElementById("value-out-of-range-warning").style.display = "none";
    
            plotHelpers.xScale = d3.scaleTime()
                .domain(dateExtent)
                .range([0, chartWidth]);

            plotHelpers.lineGenerator = d3.line()
                .defined(d => {
                    if (d.value >= metadata.range.y.min && d.value <= metadata.range.y.max) {
                        return true;
                    } else {
                        document.getElementById("value-out-of-range-warning").style.display = "block";
                        return false;
                    }
                })
                .x(d => plotHelpers.xScale(d.date))
                .y(d => plotHelpers.yScale(d.value));
    
            axes.x.call(d3.axisBottom(plotHelpers.xScale));
    
            chartLine.datum(subData)
                .attr('d', d => plotHelpers.lineGenerator(d));
    
            
            prominentFeatures = (getProminentFeatures()).filter(f => f.prominence > 1e-5);
            clearProminentFeatures();
            displayProminentFeatures(prominentFeatures, PROMINENCE_FEATURE_COUNT);
            
            if (references) {
                removeChartReferences();
                displayChartReferences();
            }
        });
    
        const dataRange = metadata.range.y.max - metadata.range.y.min;
        const stepSize = 10 ** (Math.floor(Math.log10(dataRange)) - 1);
        const valueExtent = [dataRange / metadata.range.y.max < 0.33 ? metadata.range.y.min - 0.5 * dataRange : Math.min(metadata.range.y.min - 0.5 * dataRange, 0), dataRange < 33 ? metadata.range.y.max + 0.5 * dataRange : Math.max(metadata.range.y.max + 0.5 * dataRange, 100)];
        valueExtent[0] = Math.floor(valueExtent[0] / stepSize) * stepSize;
        valueExtent[1] = Math.ceil(valueExtent[1] / stepSize) * stepSize;
        metadata.range.y.max = Math.ceil(metadata.range.y.max / stepSize) * stepSize;
        metadata.range.y.min = Math.floor(metadata.range.y.min / stepSize) * stepSize;
        noUiSlider.create(valueRangeSlider, {
            range: {
                min: -valueExtent[1],
                max: -valueExtent[0]
            },
            step: stepSize,
            margin: stepSize,
            start: [Math.ceil(metadata.range.y.max / stepSize) * stepSize, Math.floor(metadata.range.y.min / stepSize) * stepSize],
            format: {
                to: function (value) {
                    return - value.toFixed(stepSize > 1 ? 0 : -Math.round(Math.log10(stepSize)));
                },
                from: function (value) {
                    return -value;
                }
            },
            connect: true,
            orientation: 'vertical',
            behavior: 'drag',
            tooltips: true
        });
    
        valueRangeSlider.noUiSlider.on('update', (values, handle) => {
            metadata.range.y.max = values[0];
            metadata.range.y.min = values[1];

            document.getElementById("value-out-of-range-warning").style.display = "none";

            plotHelpers.yScale = d3.scaleLinear()
                .domain([values[1], values[0]])
                .range([chartHeight, 0]);
            
    
            plotHelpers.lineGenerator = d3.line()
            .defined(d => {
                if (d.value >= metadata.range.y.min && d.value <= metadata.range.y.max) {
                    return true;
                } else {
                    document.getElementById("value-out-of-range-warning").style.display = "block";
                    return false;
                }
            })
                .x(d => plotHelpers.xScale(d.date))
                .y(d => plotHelpers.yScale(d.value));
    
            axes.y.call(d3.axisLeft(plotHelpers.yScale));
    
            chartLine.datum(subData)
                .attr('d', d => plotHelpers.lineGenerator(d));
    
            
            prominentFeatures = getProminentFeatures().filter(f => f.prominence > 1e-5);
            clearProminentFeatures();
            displayProminentFeatures(prominentFeatures, PROMINENCE_FEATURE_COUNT);

            if (references) {
                removeChartReferences();
                displayChartReferences();
            }

        });
    }
}

function toggleProminent() {
    if (document.getElementById("prominent-toggle").checked) {
        showProminentFeatures();
    } else {
        hideProminentFeatures();
    }
}

function toggleReferences() {
    if (document.getElementById("references-toggle").checked) {
        showReferences();
    } else {
        hideReferences();
    }
}

function updateProminenceThreshold() {
    prominenceThreshold = document.getElementById("prominence-threshold").value;
    clearProminentFeatures();
    displayProminentFeatures(prominentFeatures, prominenceThreshold);
    if (references !== null) {
        redrawTextReferences();
        removeChartReferences();
        displayChartReferences();
    }
}

async function fetchDataMetadata(e) {
    const chartNumberList = document.getElementById("chart-number");
    const chartNumber = chartNumberList.options[chartNumberList.selectedIndex].value;
    dataId = `line${chartNumber}`
    d3.csv(`data/real-world/data/line${chartNumber}.csv`, d => {
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
        metadata = null;
        clearChart();
        initChart();
        clearText();
        references = null;
        document.getElementById("toggle-panel").style.display = "block";
        document.getElementById("view-mode-chart-location").style.display = "block";
        document.getElementById("text-container").style.display = "block";
        document.getElementById("toggle-container").style.display = "none";

    });
}

async function uploadData(e) {
    const file = e.target.files.item(0);
    dataId = file.name.slice(0, -4);
    const text = await file.text();
    processData(text);
}

function processData(text) {
    data = d3.csvParse(text).map(d => {
        const fields = Object.keys(d)
        valueField = fields[1 - fields.indexOf("Date")];

        d.date = genISODate(d.Date);
        d.value = +d[valueField];
        delete d.Date;
        delete d[valueField];

        return d;
    });
    zippedData = {'date': [], 'value': []}
    data.forEach(_datum => {
        zippedData['date'].push(_datum['date'])
        zippedData['value'].push(_datum['value'])
    });
    subData = data;
    metadata = null;
    clearChart();
    initChart();
    clearText();
    references = null;
    document.getElementById("toggle-panel").style.display = "block";
    document.getElementById("view-mode-chart-location").style.display = "block";
    document.getElementById("text-container").style.display = "block";
    document.getElementById("toggle-container").style.display = "none";


}

function toggleChartEditMode() {
    const editModeOn = document.getElementById("chart-edit-mode").checked;
    if (editModeOn) {
        const chartContainer = document.getElementById("chart-container");
        chartContainer.remove();
        chartContainer.style.textAlign = "left";
        document.getElementById("edit-mode-chart-location").appendChild(chartContainer);
        document.getElementById("ui-table").style.display = "block";
    } else {
        const chartContainer = document.getElementById("chart-container");
        chartContainer.remove();
        chartContainer.style.textAlign = "center";
        document.getElementById("view-mode-chart-location").appendChild(chartContainer);
        document.getElementById("ui-table").style.display = "none";
    }
}

function toggleEmphasisDisplayMode() {
    const ecDisplayOn = document.getElementById("ec-display-mode").checked;
    if (ecDisplayOn) {
        layoutG.style('visibility', 'visible');
        showProminentFeatures();
        showReferences();
        if (document.getElementsByClassName("highlightTextarea-container").length > 0) {
            document.getElementsByClassName("highlightTextarea-container")[0].style.display = "block";
        }
    } else {
        layoutG.style('visibility', 'hidden');
        hideProminentFeatures();
        hideReferences();
        if (document.getElementsByClassName("highlightTextarea-container").length > 0) {
            document.getElementsByClassName("highlightTextarea-container")[0].style.display = "none";
        }
    }  
}