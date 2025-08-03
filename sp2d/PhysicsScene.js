import * as Draw from "./Draw.js";
import * as Entity from "./Entity.js";
import * as Collision from "./Collision.js";
import * as Joint from "./Joint.js";
import { Vector2, VectorMath2 } from "./Vector2.js";

var MAX_ENTITY=1e4;

export var simWidth;
export var simHeight;

export var gravity = new Vector2(0, -9.8);
export var dt = 1.0 / 60;
export var worldSize = new Vector2(0, 0); // Initialize with default values
export var paused = true;
export var airResistance = 0.0;
/**  @type {Entity.Rectangle[]|Entity.Circle[]|Entity.Polygon[]} */
export var entities = [];
/** @type {Collision.Collision[]} */
export var collisions = [];
//export var maxAngularVel = 10.0;
/**
 * @type {Joint.Joint[]}
 */
export var joints = [];

var disableColl = [];
// record if allowing collisions

export var substep;

export function init(worldSize_ = null, gravity_ = new Vector2(0.0, -9.8), airResistance_=0.0) {
    simWidth = Draw.canvas.width / Draw.scale;
    simHeight = Draw.canvas.height / Draw.scale;

    gravity = gravity_;
    worldSize = worldSize_;
    airResistance = airResistance_;
    //maxAngularVel = maxAngularVel_;

    paused = false;

    entities = [];
    collisions = [];

    dt = 1.0 / 60;

    if (worldSize == null) {
        worldSize = new Vector2(simWidth, simHeight);
    }

    for (let i = 0; i < MAX_ENTITY; i++) {
        disableColl[i] = [];
        for (let j = 0; j < MAX_ENTITY; j++) {
            disableColl[i][j] = false;
        }
    }

    console.log("Simulation width:", simWidth, "Simulation height:", simHeight);
}

export function addEntity(entity) {
    entity.id = entities.length;
    entities.push(entity);
    return entity.id;
    // Passing by reference, id has been changed. 
    // But it may make some convenience.
}

export function deleteEntity(id) {
    entities[id] = entities[entities.length - 1];
    entities.pop();
    //delete an entity in O(1) time
    //console.log(entities);
}

/**
 * @param {Joint.Joint} joint
 */
export function addJoint(joint) {
    joint.id = joints.length;
    joints.push(joint);
    return joint.id;
    // Passing by reference, id has been changed.
    // But it may make some convenience.
}

export function deleteJoint(id) {
    joints[id] = joints[entities.length - 1];
    joints.pop();
    //delete an entity in O(1) time
    //console.log(entities);
}

export function addCollision(collision) {
    collisions.push(collision);
}

export function togglePause() {
    paused = !paused;
}

export function setPaused(p) {
    paused = p;
}

export function setGravity(g) {
    gravity = g;
}

export function setDt(d) {
    dt = d;
}

export function disableCollision(a,b)
{
    if(!a||!b)
        return;
    disableColl[a][b]=true;
    disableColl[b][a]=true;
}

export function setFloorCollision(restitution = 0, friction = 0) {
    var thickness = 0.5;
    var bottom = new Entity.Rectangle(
        simWidth,
        thickness,
        restitution,
        friction,
        Infinity,
        new Vector2(simWidth / 2, -thickness / 2),
        //new Vector2(simWidth/2,0),
        VectorMath2.zero(),
        0,
        0,
        "#4F4F4F"
    );
    bottom.setStatic();
    addEntity(bottom);
}

export function setWallCollision(restitution = 0, friction = 0) {
    var thickness = 0.5;
    var top = new Entity.Rectangle(
        simWidth,
        thickness,
        restitution,
        friction,
        Infinity,
        new Vector2(simWidth / 2, simHeight + thickness / 2),
        //new Vector2(simWidth/2,simHeight),
        VectorMath2.zero(),
        0,
        0,
        "#4F4F4F"
    );
    top.setStatic();
    addEntity(top);

    var bottom = new Entity.Rectangle(
        simWidth,
        thickness,
        restitution,
        friction,
        Infinity,
        new Vector2(simWidth / 2, -thickness / 2),
        //new Vector2(simWidth/2,0),
        VectorMath2.zero(),
        0,
        0,
        "#4F4F4F"
    );
    bottom.setStatic();
    addEntity(bottom);

    var left = new Entity.Rectangle(
        thickness,
        simHeight,
        restitution,
        friction,
        Infinity,
        new Vector2(-thickness / 2, simHeight / 2),
        //new Vector2(0,simHeight/2),
        VectorMath2.zero(),
        0,
        0,
        "#4F4F4F"
    );
    left.setStatic();
    addEntity(left);

    var right = new Entity.Rectangle(
        thickness,
        simHeight,
        restitution,
        friction,
        Infinity,
        //new Vector2(simWidth,simHeight/2),
        new Vector2(simWidth + thickness / 2, simHeight / 2),
        VectorMath2.zero(),
        0,
        0,
        "#4F4F4F"
    );
    right.setStatic();
    addEntity(right);
}

export function setThickWallCollision(restitution = 0,friction = 0) {
    var thickness = 0.5;
    var top = new Entity.Rectangle(
        simWidth,
        thickness,
        restitution,
        friction,
        Infinity,
        //new Vector2(simWidth/2,simHeight+thickness/2),
        new Vector2(simWidth / 2, simHeight),
        VectorMath2.zero(),
        0,
        0,
        "#4F4F4F"
    );
    top.setStatic();
    addEntity(top);

    var bottom = new Entity.Rectangle(
        simWidth,
        thickness,
        restitution,
        friction,
        Infinity,
        //new Vector2(simWidth/2,-thickness/2),
        new Vector2(simWidth / 2, 0),
        VectorMath2.zero(),
        0,
        0,
        "#4F4F4F"
    );
    bottom.setStatic();
    addEntity(bottom);

    var left = new Entity.Rectangle(
        thickness,
        simHeight,
        restitution,
        friction,
        Infinity,
        //new Vector2(-thickness/2,simHeight/2),
        new Vector2(0, simHeight / 2),
        VectorMath2.zero(),
        0,
        0,
        "#4F4F4F"
    );
    left.setStatic();
    addEntity(left);

    var right = new Entity.Rectangle(
        thickness,
        simHeight,
        restitution,
        friction,
        Infinity,
        new Vector2(simWidth, simHeight / 2),
        // new Vector2(simWidth+thickness/2,simHeight/2),
        VectorMath2.zero(),
        0,
        0,
        "#4F4F4F"
    );
    right.setStatic();
    addEntity(right);
}

export function simulate(substep_ = 1) {
    substep=substep_;
    if (paused) return;

    collisions = [];

    // substep indicates how many times we simulate in 1 frame.
    for (let k = 0; k < substep; k++) {
        for (let i = 0; i < entities.length; i++) {
            //console.log(entities[i].pos);
            var p = entities[i].pos.clone();
            //if ((p.x > 2 * simWidth || p.x < -simWidth) && (p.y < -simHeight || p.y > 2 * simHeight)) continue;
            entities[i].simulate(dt / substep, gravity);
        }

        for (let joint of joints)
        {
            joint.apply();
        }

        for (let i = 0; i < entities.length; i++) {
            for (let j = i + 1; j < entities.length; j++) {
                if(disableColl[i][j]||disableColl[j][i]) continue;
                var collision = Collision.detect(entities[i], entities[j]);
                if (collision != null) {
                    //console.log(collision);
                    //error;
                    addCollision(collision);
                }
            }
        }

        for (let i = 0; i < collisions.length; i++) {
            collisions[i].resolve();
        }
    }
}

export function draw() {
    Draw.clear();
    for (let joint of joints) {
        //if(joint.visibility) 
        joint.draw();
    }

    for (let i = 0; i < entities.length; i++) {
        entities[i].draw();
    }
}

export function drawVertex() {
    for (let e of entities) {
        if (e.type === "Rectangle" || e.type === "Polygon") {
            let v = e.getVertices();
            for (let i of v) {
                Draw.drawCircle(i, 0.007, "#FF0000");
            }
        }
        if (e.type === "Circle") {
            let x = Math.cos(e.angle);
            let y = Math.sin(e.angle);
            //console.log(e.angle);
            Draw.drawCircle((new Vector2(x, y)).times(e.radius).add(e.pos), 0.007, "#0000FF");
        }
        Draw.drawCircle(e.pos, 0.007, "#FF00FF");
    }
}

export function drawContactPoint() {
    for (let c of collisions) {
        Draw.drawCircle(c.contactPoint, 0.007, "#FFFF00");
    }
}

export function drawAnchorPoint() {
    for (let j of joints) {
        
        var rotatedAnchorA = j.anchorA.rotate(-j.angleA+j.bodyA.angle);
        var rotatedAnchorB = j.anchorB.rotate(-j.angleB+j.bodyB.angle);

        var Pa = j.bodyA.pos.add(rotatedAnchorA);
        //Absolute coordinates of unconstrained entity
        var Pb = j.bodyB.pos.add(rotatedAnchorB);

        Draw.drawCircle(Pa, 0.007, "#3f3f3f");
        Draw.drawCircle(Pb, 0.007, "#3f3f3f");
    }
}