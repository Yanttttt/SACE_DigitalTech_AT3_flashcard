import * as Entity from "./Entity.js";
import { Vector2, VectorMath2 } from "./Vector2.js";

export var canvas;
export var ctx;
export var scale;
export var div;

/**
 * @param {string} canvasID
 */
export function init(canvasID, width = window.innerWidth - 20, height = window.innerHeight - 100, div_ = 2) {
    div = div_;
    canvas = /** @type {HTMLCanvasElement} */ (document.getElementById(canvasID));
    ctx = /** @type {HTMLCanvasElement} */ (canvas).getContext("2d");
    canvas.width = width;
    canvas.height = height;

    scale = Math.min(canvas.width, canvas.height) / div;
}

export function convertX(pos) {
    return pos.x * scale;
}

export function convertY(pos) {
    return canvas.height - pos.y * scale;
}
//convert the coordinate system of physics world to that of the canvas

export function getRandomColour() {
    let r = Math.floor(Math.random() * 256).toString(16);
    let g = Math.floor(Math.random() * 256).toString(16);
    let b = Math.floor(Math.random() * 256).toString(16);

    r = r.length === 1 ? "0" + r : r;
    g = g.length === 1 ? "0" + g : g;
    b = b.length === 1 ? "0" + b : b;

    return "#" + (r + g + b).toUpperCase();
}



export function drawRectangle(pos, angle, width, length, colour) {
    ctx.fillStyle = colour;

    ctx.save();
    ctx.translate(convertX(pos), convertY(pos));
    ctx.rotate(-angle);
    ctx.fillRect(
        -scale * width / 2,
        -scale * length / 2,
        scale * width,
        scale * length
    );
    ctx.restore();
}

export function drawCircle(pos, radius, colour) {
    ctx.fillStyle = colour;

    ctx.beginPath();
    ctx.arc(
        convertX(pos),
        convertY(pos),
        scale * radius,
        0,
        2 * Math.PI
    );
    //ctx.closePath();
    ctx.fill();
}

export function drawPolygon(vertices, colour = "#FF0000") {
    if (!vertices || vertices.length < 3) return;
    // at least need 3 vertices.

    ctx.fillStyle = colour;
    ctx.beginPath();

    var v0 = vertices[0];
    ctx.moveTo(convertX(v0), convertY(v0));

    for (let i = 1; i < vertices.length; i++) {
        let v = vertices[i];
        ctx.lineTo(convertX(v), convertY(v));
    }

    ctx.closePath();
    ctx.fill();
}

export function drawSpring(start, end, numCoils = 5, amplitude = 0, colour = "#000000") {
    var points = [];
    var direction = end.subtract(start).normalise();
    var perpendicular = new Vector2(-direction.y, direction.x);

    var length = start.subtract(end).length();
    //var step = length / (numCoils * 10);

    for (let i = 0; i <= numCoils * 4; i++) {
        let t = i / (numCoils * 4);
        let pos = start.add(direction.times(t * length));
        let offset = perpendicular.times(
            Math.sin(t * Math.PI * 2 * numCoils) * amplitude);
        points.push(pos.add(offset));
    }

    ctx.strokeStyle = colour;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(convertX(points[0]), convertY(points[0]));
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(convertX(points[i]), convertY(points[i]));
    }
    ctx.stroke();
}

export function drawText(text, pos, colour = "#000000", fontSize = 16) {
    ctx.fillStyle = colour;
    ctx.font = `${fontSize}px Arial`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, convertX(pos), convertY(pos));
}

export function drawLine(start, end, colour = "#000000", width = 1) {
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(convertX(start), convertY(start));
    ctx.lineTo(convertX(end), convertY(end));
    ctx.stroke();
}

export function clear() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}