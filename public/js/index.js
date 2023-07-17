"use strict"

window.onload = init;

var userId = "admin";
var sessionId;

function init() {
    const queryVars = readQuery();

    initDropZone();
    drawDefaultChart("real-world|line013");
    initText();
    sessionId = generateRandomId();
    userId = generateRandomUserId();
}
