"use strict"

var highlightOrdering;
var sentenceBoundingBoxes = [];
var phraseBoundingBoxes = {};
var currMouseSentence = null;
var currMousePhrase = null;
var references = null;

function initText() {
    const inputTextarea = document.getElementById("input-text");
    inputTextarea.addEventListener('keypress', updateReferences); // Every character written
    inputTextarea.addEventListener('input', updateReferences); // Other input events
    inputTextarea.addEventListener('paste', updateReferences); // Clipboard actions
    inputTextarea.addEventListener('cut', updateReferences);
    inputTextarea.addEventListener('select', updateReferences); // Some browsers support this event
    inputTextarea.addEventListener('selectstart', updateReferences); // Some browsers support this event
}

var lastSentence = "";
var editsSinceLastUpdate = false;
var lastCursorSentences = [];
async function updateReferences(e) {
    const inputTextarea = document.getElementById("input-text");
    const inputText = inputTextarea.value;
    if (references === null) {
        if (e.keyCode == 13 && e.shiftKey) {
            e.preventDefault();
            lastSentence = inputText;
            inputTextarea.disabled = true;
            references = await getReferences(data, inputText);
            addReferenceNumbering();
            displayTextReferences();
            inputTextarea.focus();
            removeChartReferences();
            displayChartReferences();
            inputTextarea.disabled = false;
        }
        return;
    }

    if (e.keyCode == 13 && e.shiftKey) {
        e.preventDefault();
        inputTextarea.disabled = true;
        references = await getReferences(data, inputText);
        addReferenceNumbering();
        redrawTextReferences();
        inputTextarea.focus();
        removeChartReferences();
        displayChartReferences();
        inputTextarea.disabled = false;
        return;
    }

    const cursorStart = document.getElementById('input-text').selectionStart;
    const cursorEnd = document.getElementById('input-text').selectionEnd;
    const currCursorSentences = [];
    references.forEach((sentenceReferences, sentenceIdx) =>{
        if (sentenceReferences.charIdx[0] >= cursorStart && sentenceReferences.charIdx[1] <= cursorEnd) {
            currCursorSentences.push(sentenceIdx);
        }
    });

    if (inputText === lastSentence) {
        return;
    }

    
    editsSinceLastUpdate = true;
    const shift = inputText.length - lastSentence.length;
    var diffStartIdx = 0;
    while (inputText[diffStartIdx] === lastSentence[diffStartIdx]) {
        diffStartIdx += 1;
    }
    var diffEndIdx = 1;
    while (inputText[inputText.length - diffEndIdx] === lastSentence[lastSentence.length - diffEndIdx]) {
        diffEndIdx += 1;
    }
    diffEndIdx = lastSentence.length - diffEndIdx;
    var delCharsIdx = {start: diffStartIdx, end: diffEndIdx + 1};
    if (diffEndIdx < diffStartIdx) {
        delCharsIdx = null;
    }
    shiftReferences(cursorEnd, shift, delCharsIdx);
    redrawTextReferences();
    inputTextarea.focus();
    lastSentence = inputText;
    removeChartReferences();
    displayChartReferences();
}

async function checkSentenceEnd() {
    const inputTextarea = document.getElementById("input-text");
    const inputText = inputTextarea.value;
    if (inputText[inputText.length - 1] === '.') {
        references = await getReferences(data, inputText);
        addReferenceNumbering(references);
        displayTextReferences();
        inputTextarea.focus();
        removeChartReferences();
        displayChartReferences();
    }
}

var nextReferenceNumbering = 0;
var unusedReferenceNumbering = [];
// For coloring purposes.
function addReferenceNumbering() {
    references.forEach(sentenceReferences => {
        sentenceReferences.references.forEach(reference => {
            if (unusedReferenceNumbering.length > 0) {
                reference.numbering = unusedReferenceNumbering.shift();
                return;
            }
            reference.numbering = nextReferenceNumbering;
            nextReferenceNumbering++;
        })
    });
}

function displayTextReferences() {
    const toHighlight = [];
    highlightOrdering = [];
    references.forEach(sentenceReferences => {
        toHighlight.push({
            ranges: [sentenceReferences.charIdx]
        });
        highlightOrdering.push(`sentence-${sentenceReferences.sentenceIdx}`);
    });
    $("#input-text").highlightTextarea({
        ranges: toHighlight
    });
    modifyHighlightTextarea();
}

function redrawTextReferences() {
    clearTextReferences();
    const toHighlight = [];
    highlightOrdering = [];
    references.forEach(sentenceReferences => {
        toHighlight.push({
            ranges: [sentenceReferences.charIdx]
        });
        highlightOrdering.push(`sentence-${sentenceReferences.sentenceIdx}`);
    });
    $("#input-text").highlightTextarea({
        ranges: toHighlight
    });
    modifyHighlightTextarea();    
}

function modifyHighlightTextarea() {
    const marks = document.getElementById("text-container").getElementsByTagName('mark');
    for (let markIdx = 0; markIdx < marks.length; markIdx++) {
        let mark = marks[markIdx];
        mark.id = highlightOrdering[markIdx];
        mark.classList.add("highlight-default");
    }

    const sentenceMarks = document.querySelectorAll("mark[id^='sentence-']");

    sentenceBoundingBoxes = [];
    for (let sentenceMarkIdx = 0; sentenceMarkIdx < sentenceMarks.length; sentenceMarkIdx++) {
        let sentenceMark = sentenceMarks[sentenceMarkIdx];
        sentenceBoundingBoxes.push(getTextBoundingBoxes(sentenceMark));
    }

    document.getElementById("text-container").onmousemove = (event) => {
        const sentenceIdxs = findMarksByMouseCoord({x: event.pageX, y: event.pageY}, sentenceBoundingBoxes).map(idx => {
            return parseInt(sentenceMarks[idx].id.substring(9));
        });
        if (currMouseSentence === null && sentenceIdxs.length > 0) {
            currMouseSentence = sentenceIdxs[0];
        } else if (currMouseSentence !== null) {
            if (sentenceIdxs.length === 0) {
                currMouseSentence = null;
            } else if (currMouseSentence !== sentenceIdxs[0]) {
                currMouseSentence = sentenceIdxs[0];
            }
        }

        if (currMouseSentence !== null) {
            let phraseIdxs = findPhrasesByMouseCoord({x: event.pageX, y: event.pageY}, phraseBoundingBoxes[currMouseSentence]).map(idx => {
                return references.filter(r => r.sentenceIdx === currMouseSentence)[0].references[idx].referenceIdx;
            });
            if (currMousePhrase === null && phraseIdxs.length > 0) {
                modifyHighlightingReferences('highlight-deemph');
                modifyHighlightingProminentFeatures("highlight-deemph");
                modifyHighlightingMark('highlight-deemph');
                phraseIdxs.forEach(phraseIdx => {
                    const idSplit = phraseIdx.split('-');
                    modifyHighlightingReferences('highlight-emph', idSplit[0], idSplit[1]);
                    modifyHighlightingMark('highlight-emph', idSplit[0], idSplit[1])
                });
                hideXTicks();
                currMousePhrase = phraseIdxs;
            } else if (currMousePhrase !== null) {
                if (phraseIdxs.length === 0) {
                    modifyHighlightingReferences('highlight-default');
                    modifyHighlightingProminentFeatures("highlight-default");
                    modifyHighlightingMark('highlight-default');
                    showXTicks();
                    currMousePhrase = null;
                } else if (JSON.stringify(currMousePhrase) === JSON.stringify(phraseIdxs)) {
                    // Note: the arrays are absolutely ordered, so this is sufficient to check for equality.
                    modifyHighlightingReferences('highlight-deemph');
                    modifyHighlightingProminentFeatures("highlight-deemph");
                    modifyHighlightingMark('highlight-deemph');
                    phraseIdxs.forEach(phraseIdx => {
                        const idSplit = phraseIdx.split('-');
                        modifyHighlightingReferences('highlight-emph', idSplit[0], idSplit[1]);
                        modifyHighlightingMark('highlight-emph', idSplit[0], idSplit[1])
                    });
                    hideXTicks();
                    currMousePhrase = phraseIdxs;
                }
            }
        } else if (currMousePhrase !== null) {
            modifyHighlightingReferences('highlight-default');
            modifyHighlightingProminentFeatures("highlight-default");
            modifyHighlightingMark('highlight-default');
            showXTicks();
            currMousePhrase = null;
        }
    }

    var highlightIdx = 0;
    references.forEach(sentenceReferences => {
        const currSentence = document.getElementById(highlightOrdering[highlightIdx]);
        const sentenceFragments = [];
        const referenceIds = [];
        sentenceReferences.references.forEach(reference => {
            referenceIds.push(`reference-${reference.referenceIdx}`)
            if (reference.factCheck) {
                reference.text.time.forEach(t => {
                    sentenceFragments.push({range: t, classList: [`reference-${reference.referenceIdx}`, `reference-${reference.referenceIdx}-text`], coloring: [reference.numbering]});
                })
                sentenceFragments.push({range: reference.text.feature, classList: [`reference-${reference.referenceIdx}`, `reference-${reference.referenceIdx}-text`], coloring: [reference.numbering]});
            } else {
                reference.text.time.forEach(t => {
                    sentenceFragments.push({range: t, classList: [`reference-${reference.referenceIdx}`, `reference-${reference.referenceIdx}-text fact-false`], coloring: [reference.numbering]});
                })
                sentenceFragments.push({range: reference.text.feature, classList: [`reference-${reference.referenceIdx}`, `reference-${reference.referenceIdx}-text`, "fact-false"], coloring: [reference.numbering]});
            }
        });

        // Reverse-ordering so that we can traverse through each.
        sentenceFragments.sort(function(a, b) {
            if (a.range[0] == b.range[0]) {
                return b.range[1] - a.range[1];
            }
            return b.range[0] - a.range[0];
        })

        const clusteredSentenceFragments = []
        var prevFragment = null;
        sentenceFragments.forEach(sentenceFragment => {
            if (prevFragment !== null) {
                if (prevFragment.range[0] === sentenceFragment.range[0] && prevFragment.range[1] === sentenceFragment.range[1]) {
                    prevFragment.classList.push(sentenceFragment.classList[0]);
                    prevFragment.classList.push(sentenceFragment.classList[1]);

                    prevFragment.coloring.push(sentenceFragment.coloring[0])
                } else if (prevFragment.range[0] < sentenceFragment.range[1]) {
                    $.error("Ranges overlap!");
                } else {
                    clusteredSentenceFragments.push(sentenceFragment);
                    prevFragment = sentenceFragment;
                }
            } else {
                clusteredSentenceFragments.push(sentenceFragment);
                prevFragment = sentenceFragment;
            }
        });

        const originalMarkContents = currSentence.textContent;
        var newMarkContents = "";
        var lastCharIdx = originalMarkContents.length;
        clusteredSentenceFragments.forEach(sentenceFragment => {
            newMarkContents = `<span class="${sentenceFragment.classList.join(' ')} highlight-default coloring-${sentenceFragment.coloring.join('-')}">${originalMarkContents.substring(sentenceFragment.range[0] - sentenceReferences.charIdx[0], sentenceFragment.range[1] - sentenceReferences.charIdx[0])}</span>${originalMarkContents.substring(sentenceFragment.range[1] - sentenceReferences.charIdx[0], lastCharIdx)}` + newMarkContents;
            lastCharIdx = sentenceFragment.range[0] - sentenceReferences.charIdx[0];
        });
        newMarkContents = originalMarkContents.substring(0, lastCharIdx) + newMarkContents;
        currSentence.innerHTML = newMarkContents;

        const sentencePhrases = [];
        referenceIds.forEach(referenceId => {
            var boundingBoxes = [];
            const textReferences = document.getElementsByClassName(`${referenceId}-text`);
            for (let textReferenceIdx = 0; textReferenceIdx < textReferences.length; textReferenceIdx++) {
                boundingBoxes = boundingBoxes.concat(getTextBoundingBoxes(textReferences[textReferenceIdx]));
            }
            sentencePhrases.push(boundingBoxes);
        });
        phraseBoundingBoxes[highlightOrdering[highlightIdx].substring(9)] = sentencePhrases;
        

        highlightIdx += 1;
    });
}



function findMarksByMouseCoord(mouse, candidates) {
    const marks = [];
    candidates.forEach((candidate, candidateIdx) => {
        for (let boxIdx = 0; boxIdx < candidate.length; boxIdx++) {
            let box = candidate[boxIdx];
            if (mouse.x >= box.left && mouse.x < box.right && mouse.y >= box.top && mouse.y < box.bottom) {
                marks.push(candidateIdx);
                return;
            }
        }
    });
    return marks;
}

function findPhrasesByMouseCoord(mouse, candidates) {
    const phrases = [];
    candidates.forEach((candidate, candidateIdx) => {
        for (let boxIdx = 0; boxIdx < candidate.length; boxIdx++) {
            let box = candidate[boxIdx];
            if (mouse.x >= box.left && mouse.x < box.right && mouse.y >= box.top && mouse.y < box.bottom) {
                phrases.push(candidateIdx);
                return;
            }
        }
    });
    return phrases;    
}

function getTextBoundingBoxes(elem) {
    // Because text can wrap, need multiple bounding boxes to capture the area of a element including text (e.g., mark, span).
    // Modification of the code from: https://stackoverflow.com/questions/62749538/find-how-the-lines-are-split-in-a-word-wrapped-element
    const node = elem.firstChild;
    const text = elem.textContent;
    const range = document.createRange();
    range.selectNodeContents(elem);
    range.setStart(node, 0);
    range.setEnd(node, 1);
    const boundingBoxes = [];
    
    var rowHeight = range.getBoundingClientRect().height;
    for (let charIdx = 0; charIdx < text.length - 2; charIdx++) {
        range.setEnd(node, charIdx);
        let currHeight = range.getBoundingClientRect().height;
        if (currHeight > rowHeight) {
            range.setEnd(node, charIdx - 1);
            boundingBoxes.push(getPageRect(range));

            range.setStart(node, charIdx - 1);
        }
    }
    range.setEnd(node, text.length - 1);
    boundingBoxes.push(getPageRect(range));
    return boundingBoxes;
}

function modifyHighlightingMark(newHighlight, sentenceIdx = null, referenceIdx = null) {
    var identifier;
    var marks;
    var spans;
    if (sentenceIdx === null) {
        identifier = "'sentence-'";
        marks = document.querySelectorAll(`mark[id^=${identifier}]`);
        spans = document.getElementById("text-container").getElementsByTagName('span');
        for (let markIdx = 0; markIdx < marks.length; markIdx++) {
            let mark = marks[markIdx];
            mark.classList.remove("highlight-default", "highlight-emph", "highlight-deemph", "highlight-outfocus");
            mark.classList.add(newHighlight);
        }
    } else if (referenceIdx === null) {
        identifier = `'sentence-${sentenceIdx}'`;
        marks = document.querySelectorAll(`mark[id^=${identifier}]`);
        spans = marks[0].getElementsByTagName('span');
        for (let markIdx = 0; markIdx < marks.length; markIdx++) {
            let mark = marks[markIdx];
            mark.classList.remove("highlight-default", "highlight-emph", "highlight-deemph", "highlight-outfocus");
            mark.classList.add(newHighlight);
        }
    } else {
        spans = document.getElementById("text-container").getElementsByClassName(`reference-${sentenceIdx}-${referenceIdx}`);
    }
    for (let spanIdx = 0; spanIdx < spans.length; spanIdx++) {
        let span = spans[spanIdx];
        span.classList.remove("highlight-default", "highlight-emph", "highlight-deemph", "highlight-outfocus");
        span.classList.add(newHighlight);
    }
}

function shiftReferences(cursorIdx, shift, delCharIdxs) {
    const sentencesToRemove = [];
    var mergeStart = null;
    var mergeEnd = null;
    var needRerender = true;
    references.forEach((sentenceReferences, sentenceIdx) => {
        if (delCharIdxs !== null) {
            if (delCharIdxs.start > sentenceReferences.charIdx[1]) {
                return;
            } if (delCharIdxs.start <= sentenceReferences.charIdx[0] && delCharIdxs.end >= sentenceReferences.charIdx[1]) {
                sentencesToRemove.push(sentenceIdx);
                return;
            } else if (delCharIdxs.start > sentenceReferences.charIdx[0] && delCharIdxs.start <= sentenceReferences.charIdx[1] && delCharIdxs.end > sentenceReferences.charIdx[1]) {
                sentenceReferences.charIdx[1] = delCharIdxs.start;
                mergeStart = sentenceIdx;
            } else if (delCharIdxs.start < sentenceReferences.charIdx[0] && delCharIdxs.end >= sentenceReferences.charIdx[0] && delCharIdxs.end < sentenceReferences.charIdx[1]) {
                sentenceReferences.charIdx[0] = delCharIdxs.end;
                if (mergeStart !== null) {
                    sentencesToRemove.push(sentenceIdx);
                }
                mergeEnd = sentenceIdx;
            } else if (delCharIdxs.end < sentenceReferences.charIdx[1]) {
                sentenceReferences.charIdx[1] += shift;
                if (delCharIdxs.end < sentenceReferences.charIdx[0]) {
                    sentenceReferences.charIdx[0] += shift;
                }
            }
        } else {
            if (cursorIdx - shift > sentenceReferences.charIdx[1]) {
                return;
            }
            if (cursorIdx - shift < sentenceReferences.charIdx[0]) {
                sentenceReferences.charIdx[0] += shift;
            }
            sentenceReferences.charIdx[1] += shift;
        }
        const referencesToRemove = [];
        sentenceReferences.references.forEach((reference, referenceIdx) => {

            if (delCharIdxs !== null) {
                for (let tIdx = 0; tIdx < reference.text.time.length; tIdx++) {
                    let t = reference.text.time[tIdx];
                    if (delCharIdxs.start <= t[0] && delCharIdxs.end >= t[1]) {
                        // Case: Complete inclusion within removed area
                        referencesToRemove.push(referenceIdx);
                        return;
                    } else if (delCharIdxs.start >= t[0] && delCharIdxs.start < t[1]) {
                        // Case: Start overlaps
                        referencesToRemove.push(referenceIdx);
                        return;
                    } else if (delCharIdxs.end > t[0] && delCharIdxs.end <= t[1]) {
                        // Case: End overlaps
                        referencesToRemove.push(referenceIdx);
                        return;
                    } else if (delCharIdxs.end <= t[0]) {
                        t[0] += shift;
                        t[1] += shift;
                    }
                }
                if (delCharIdxs.start <= reference.text.feature[0] && delCharIdxs.end >= reference.text.feature[1]) {
                    // Case: Complete inclusion within removed area
                    referencesToRemove.push(referenceIdx);
                    return;
                } else if (delCharIdxs.start >= reference.text.feature[0] && delCharIdxs.start < reference.text.feature[1]) {
                    // Case: Start overlaps
                    referencesToRemove.push(referenceIdx);
                    return;
                } else if (delCharIdxs.end > reference.text.feature[0] && delCharIdxs.end <= reference.text.feature[1]) {
                    // Case: End overlaps
                    referencesToRemove.push(referenceIdx);
                    return;
                } else if (delCharIdxs.end <= reference.text.feature[0] ) {
                    reference.text.feature[0] += shift;
                    reference.text.feature[1] += shift;
                }  
            } else {
                for (let tIdx = 0; tIdx < reference.text.time.length; tIdx++) {
                    let t = reference.text.time[tIdx];
                    if (cursorIdx - shift > t[0] && cursorIdx - shift < t[1]) {
                        referencesToRemove.push(referenceIdx);
                        return;
                    } else if (cursorIdx - shift <= t[0]) {
                        t[0] += shift;
                        t[1] += shift;
                    }
                }

                if (cursorIdx - shift > reference.text.feature[0] && cursorIdx - shift < reference.text.feature[1]) {
                    referencesToRemove.push(referenceIdx);
                    return;
                } else if (cursorIdx - shift <= reference.text.feature[0]) {
                    reference.text.feature[0] += shift;
                    reference.text.feature[1] += shift;
                } 
            }
        });
        referencesToRemove.reverse().forEach(referenceIdx => {
            const referenceIdxSplit = sentenceReferences.references[referenceIdx].referenceIdx.split("-");
            removeChartReferences(parseInt(referenceIdxSplit[0]), parseInt(referenceIdxSplit[1]));
            sentenceReferences.references.splice(referenceIdx, 1);
        });
    });
    if (mergeStart !== null && mergeEnd !== null) {
        needRerender = true;
        references[mergeStart].charIdx[1] = references[mergeEnd].charIdx[1];
        let lastUsedIdx = -1;
        references[mergeStart].references.forEach(r => {
            lastUsedIdx = Math.max(lastUsedIdx, parseInt(r.referenceIdx.split("-")[1]));
        });
        references[mergeEnd].references.forEach(r => {
            r.referenceIdx = `${references[mergeStart].sentenceIdx}-${lastUsedIdx + 1}`;
            lastUsedIdx += 1;
            references[mergeStart].references.push(r);
        });
    }
    sentencesToRemove.reverse().forEach(sentenceIdx => {
        removeChartReferences(references[sentenceIdx].sentenceIdx);
        references.splice(sentenceIdx, 1);
    });
    return needRerender;
}

function clearTextReferences() {
    $("#input-text").highlightTextarea("destroy");
    nextReferenceNumbering = 0;
    unusedReferenceNumbering = [];
}

function clearText() {
    document.getElementById("input-text").value = "";
    clearTextReferences();
}