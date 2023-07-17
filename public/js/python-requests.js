"use strict"

const sentFiles = [];
const featuresCache = [];

function getProminentFeatures() {
    var dataToSend = (sentFiles.indexOf(dataId) > -1 ? null : dateToString(data));
    if (sentFiles.indexOf(dataId) === -1) {
        sentFiles.push(dataId);
    }

    var features = _getProminentFeatures(zippedData, metadata);
    return features;
}

function getReferences(data, text) {        
    return new Promise((resolve, reject) => {
        fetch(`${urls[URL_OPTION]}:${PORTNUM}/get-references`, {
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' }, 
            body: JSON.stringify({userId: userId, sessionId: sessionId, dataId: dataId, data: dateToString(data), text: text})
            })
            .then(res => res.json())
            .then(data => { 
                console.log(data)
                resolve(data);
            });
        });
}

function dateToString(data) {
    return data.map(d => ({date: `${d.date.getUTCFullYear()}-${`0${d.date.getUTCMonth() + 1}`.slice(-2)}-${`0${d.date.getUTCDate()}`.slice(-2)}`, value: d.value}))
}
