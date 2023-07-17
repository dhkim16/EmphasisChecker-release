// This is an implementation of Ramer-Douglas-Peucker by [Christopher Henn](https://observablehq.com/@chnn/running-ramer-douglas-peucker-on-typed-arrays)

function rdp(xs, ys, epsilon) {
    const keep = new Uint8Array(xs.length);
    const sqEpsilon = epsilon * epsilon;
  
    keep[0] = 1;
    keep[keep.length - 1] = 1;
  
    simplifyDouglasPeuckerHelper(xs, ys, sqEpsilon, 0, keep.length - 1, keep);
  
    let resultLength = 0;
  
    // TODO: Would closing over a resultLength variable in the helper be faster?
    for (let j = 0; j < keep.length; j++) {
      if (keep[j] === 1) {
        resultLength++;
      }
    }
  
    const xsResult = new Array(resultLength);
    const ysResult = new Array(resultLength);
  
    let i = 0;
  
    for (let j = 0; j < keep.length; j++) {
      if (keep[j] === 1) {
        xsResult[i] = xs[j];
        ysResult[i] = ys[j];
        i++;
      }
    }
  
    return [xsResult, ysResult];
  }

function simplifyDouglasPeuckerHelper(xs, ys, epsilonSq, i0, i1, keep) {
    const x0 = xs[i0];
    const y0 = ys[i0];
    const x1 = xs[i1];
    const y1 = ys[i1];
  
    let maxIndex = 0;
    let maxDist = -1;
  
    for (let i = i0 + 1; i < i1; i++) {
      const sqDist = sqSegmentDist(x0, y0, x1, y1, xs[i], ys[i]);
  
      if (sqDist > maxDist) {
        maxIndex = i;
        maxDist = sqDist;
      }
    }
  
    if (maxDist > epsilonSq) {
      keep[maxIndex] = 1;
  
      if (maxIndex - i0 > 1) {
        simplifyDouglasPeuckerHelper(xs, ys, epsilonSq, i0, maxIndex, keep);
      }
  
      if (i1 - maxIndex > 1) {
        simplifyDouglasPeuckerHelper(xs, ys, epsilonSq, maxIndex, i1, keep);
      }
    }
  }

// Shortest distance from (x2, y2) to the line segment between (x0, y0) and (x1, y1)
//
// Adapted from the [Simplify.js](https://mourner.github.io/simplify-js/) library.
function sqSegmentDist(x0, y0, x1, y1, x2, y2) {
    let x = x0;
    let y = y0;
    let dx = x1 - x0;
    let dy = y1 - y0;
  
    if (dx !== 0 || dy !== 0) {
      var t = ((x2 - x) * dx + (y2 - y) * dy) / (dx * dx + dy * dy);
  
      if (t > 1) {
        x = x1;
        y = y1;
      } else if (t > 0) {
        x += dx * t;
        y += dy * t;
      }
    }
  
    dx = x2 - x;
    dy = y2 - y;
  
    return dx * dx + dy * dy;
}