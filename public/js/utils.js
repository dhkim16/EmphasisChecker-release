"use strict"
function isClose(a, b, rel_tol=1e-09, abs_tol=0.0) {
    return Math.abs(a - b) <= Math.max(rel_tol * Math.max(Math.abs(a), Math.abs(b)), abs_tol);
  }


function getPageRect(el) {
    // Modification of code from: https://stackoverflow.com/questions/25630035/javascript-getboundingclientrect-changes-while-scrolling
    var
        left = 0,
        top = 0,
        width = 0,
        height = 0,
        offsetBase = getPageRect.offsetBase;
    if (!offsetBase && document.body) {
        offsetBase = getPageRect.offsetBase = document.createElement('div');
        offsetBase.style.cssText = 'position:absolute;left:0;top:0';
        document.body.appendChild(offsetBase);
    }
    var boundingRect = el.getBoundingClientRect();
    var baseRect = offsetBase.getBoundingClientRect();
    left = boundingRect.left - baseRect.left;
    top = boundingRect.top - baseRect.top;
    width = boundingRect.right - boundingRect.left;
    height = boundingRect.bottom - boundingRect.top;
    return {
        left: left,
        top: top,
        width: width,
        height: height,
        right: left + width,
        bottom: top + height
    };
}

function generateRandomId() {
    return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
}


function readQuery() {
    var queryVars = {};
    var query = window.location.search.substring(1);
    var vars = query.split("&");
    for (var i = 0; i < vars.length; i++) {
        var pair = vars[i].split("+").join(" ").split("=");
        if (typeof queryVars[pair[0]] === "undefined") {
            queryVars[pair[0]] = decodeURIComponent(pair[1]);
        } else if (typeof queryVars[pair[0]] === "string") {
            var arr = [queryVars[pair[0]],decodeURIComponent(pair[1])];
            queryVars[pair[0]] = arr;
        } else {
            queryVars[pair[0]].push(decodeURIComponent(pair[1]));
        }
    } 
    return queryVars;
}

const idAdjectives = ["cozy", "drowzy", "upbeat", "grumpy", "happy", "jumpy", "excited", "dancing", "lying", "sitting", "standing", "running", "jumping", "crawling", "walking", "swimming", "flying", "singing", "dancing", "playing", "sleeping", "eating", "drinking", "working", "reading", "writing", "drawing", "painting", "screaming", "laughing", "crying", "smiling", "frowning", "winking", "talking", "whispering", "shouting", "screaming", "yelling", "singing", "humming", "tapping", "clapping", "cheering", "coughing", "sneezing", "snoring", "sniffing", "barking", "meowing", "roaring", "hissing", "squeaking"];
const idAdjectives2 = ["Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Pink", "Brown", "Grey", "Black", "White", "Silver", "Gold", "Bronze", "Copper", "Platinum", "Ruby", "Sapphire", "Emerald", "Diamond", "Amethyst", "Topaz", "Opal", "Pearl", "Onyx", "Jade", "Turquoise", "Lapis", "Obsidian"];
const idNouns = ["Squirrel", "Whale", "Dog", "Cat", "Mouse", "Bat", "Koala", "Giraffe", "Elephant", "Lion", "Tiger", "Bear", "Panda", "Monkey", "Horse", "Cow", "Sheep", "Goat", "Chicken", "Duck", "Goose", "Frog", "Toad", "Snake", "Lizard", "Turtle", "Fish", "Shark", "Octopus", "Dolphin", "Seal", "Otter", "Beaver", "Rabbit", "Hamster", "Parrot", "Penguin", "Owl", "Eagle", "Hawk", "Sparrow", "Crow", "Seagull", "Pelican", "Crane", "Flamingo", "Swan"];

function generateRandomUserId() {
    return `${idAdjectives[Math.floor(Math.random() * idAdjectives.length)]}${idAdjectives2[Math.floor(Math.random() * idAdjectives2.length)]}${idNouns[Math.floor(Math.random() * idNouns.length)]}${Math.floor(Math.random() * 10000).toString().padStart(4, '0')}`;
}

function genISODate(datestr) {
    if (datestr instanceof Date) {
        return datestr;
    }
    if (typeof datestr == "number") {
        return new Date(datestr);
    }
    return new Date(`${datestr}Z`)
}