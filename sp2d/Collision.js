import { Vector2, VectorMath2 } from "./Vector2.js";
import * as Entity from "./Entity.js";
import * as PhysicsScene from "./PhysicsScene.js";
import * as Draw from "./Draw.js";

export class Collision {
    /**
     * @param {any} entity1
     * @param {any} entity2
     * @param {Vector2} normal
     * @param {number} depth
     * @param {Vector2} contactPoint
     */
    constructor(entity1, entity2, normal, depth, contactPoint) {
        this.entity1 = entity1; //nested object, parameter passed by reference
        this.entity2 = entity2;
        this.normal = normal; //supposed to be e1 -> e2
        this.depth = depth;
        this.contactPoint = contactPoint;
    }

    resolve() {
        var corrFactor = 0.3/PhysicsScene.substep;
        //var corrFactor =0;
        //console.log(corrFactor);
        // change corrFactor here

        var a = this.entity1;
        var b = this.entity2;
        var normal = this.normal.clone();
        var depth = this.depth;
        var P = this.contactPoint;

        var e = Math.min(a.restitution, b.restitution);

        if (a.mass === Infinity && b.mass === Infinity) {
            return;
        }// 2 static entities
        // console.log(this.contactPoint);

        var r_ap = P.subtract(a.pos);
        var r_bp = P.subtract(b.pos);

        var v_a = a.vel.add(r_ap.perpendicular(), a.angularVel);
        var v_b = b.vel.add(r_bp.perpendicular(), b.angularVel);

        //console.log(b.angularVel);

        var v_rel = v_b.subtract(v_a);
        var v_rel_n = v_rel.dot(normal);

        if(v_rel_n>0)
        {
            return;
        }//seperating

        var r_ap_cross_n = r_ap.cross(normal);
        var r_bp_cross_n = r_bp.cross(normal);

        var denom = 0;
        if (a.mass !== Infinity) {
            denom += a.massInv + (r_ap_cross_n * r_ap_cross_n) * a.inertiaInv;
        }
        // console.log(denom);
        // error;
        if (b.mass !== Infinity) {
            denom += b.massInv + (r_bp_cross_n * r_bp_cross_n) * b.inertiaInv;

            //console.log(b.massInv,r_bp_cross_n,b.inertiaInv);
            // error;
        }

        // console.log(denom);
        // error;

        if (denom === 0) {
            console.log("denom == 0");
            return;
        }
        denom = Math.max(denom, 1e-20);

        var tangent = normal.perpendicular();
        var v_rel_t = v_rel.dot(tangent); //tangent vel
        var Jf = -v_rel_t / denom; //friction impulse

        //console.log(Jf);

        var mu = (a.friction + b.friction) / 2;
        //console.log(mu);

        var J = -(1 + e) * v_rel_n / denom;
        Jf = Math.max(-mu * Math.abs(J), Math.min(mu * Math.abs(J), Jf)); 
        
        var impulse = normal.times(J);
        var frictionImpulse = tangent.times(Jf);

        //console.log(tangent,Jf,tangent.times(Jf));

        if (a.mass !== Infinity) {
            a.vel.addEqual(impulse.times(-a.massInv));
            a.angularVel -= r_ap.cross(impulse) * a.inertiaInv;

            a.vel.addEqual(frictionImpulse.times(-a.massInv));
            a.angularVel -= r_ap.cross(frictionImpulse) * a.inertiaInv;
        }
        if (b.mass !== Infinity) {
            b.vel.addEqual(impulse.times(b.massInv));
            b.angularVel += r_bp.cross(impulse) * b.inertiaInv;

            b.vel.addEqual(frictionImpulse.times(b.massInv));
            b.angularVel += r_bp.cross(frictionImpulse) * b.inertiaInv;
        }

        // if (a.mass !== Infinity) {
        //     a.angularVelocity = Math.min(
        //         Math.max(a.angularVel, -PhysicsScene.maxAngularVel),
        //         PhysicsScene.maxAngularVel
        //     );
        // }
        // if (b.mass !== Infinity) {
        //     b.angularVelocity = Math.min(
        //         Math.max(b.angularVel, -PhysicsScene.maxAngularVel),
        //         PhysicsScene.maxAngularVel
        //     );
        // }

        //avoid penetration
        //if (depth < 0.0005) depth = 0;

        var totalMassInv = (a.mass !== Infinity ? a.massInv : 0) + (b.mass !== Infinity ? b.massInv : 0);
        if (totalMassInv > 0) {
            var corr = normal.times(depth * corrFactor / totalMassInv);
            if (a.mass !== Infinity) {
                a.pos.addEqual(corr.times(-a.massInv));
            }
            if (b.mass !== Infinity) {
                b.pos.addEqual(corr.times(b.massInv));
            }
        }
    }
}

export function detect(entity1, entity2) {
    if ((entity1.type === "Circle"||entity1.type === "Particle") && (entity2.type === "Circle"||entity2.type === "Particle")) {
        return detectCircles(entity1, entity2);
    }
    if (
        (entity1.type === "Rectangle" || entity1.type === "Polygon") && 
        (entity2.type === "Rectangle" || entity2.type === "Polygon")
    ) {
        //console.log("polygons!");
        return detectPolygons(entity1, entity2);
    }
    if (((entity1.type === "Circle"||entity1.type === "Particle") && 
        (entity2.type === "Rectangle" || entity2.type === "Polygon"))) {
        return detectCirclePolygon(entity1, entity2);
    }
    if ((entity1.type === "Rectangle" || entity1.type === "Polygon") && 
    (entity2.type === "Circle"||entity2.type === "Particle")) {
        return detectCirclePolygon(entity2, entity1);
    }
}

/**
 * @param {Entity.Circle} c1
 * @param {Entity.Circle} c2
 */
export function detectCircles(c1, c2) {
    var dist = VectorMath2.subtract(c2.pos, c1.pos);
    if (dist.length() < c1.radius + c2.radius) {
        var depth = c1.radius + c2.radius - dist.length();
        var normal = dist.normalise();
        return new Collision(
            c1,
            c2,
            normal,
            depth,
            normal.times(c1.radius).add(c1.pos)
        );
    }
}

/**
 * @param {Entity.Circle} c
 * @param {Entity.Rectangle} r
 */
export function detectCirclePolygon(c, r) {

    var normal = VectorMath2.zero();
    var depth = Number.MAX_VALUE;

    var axis = VectorMath2.zero()
    var overlap = 0;

    for (let i = 0; i < r.vertices.length; i++) {
        let va = r.vertices[i];
        let vb = r.vertices[(i + 1) % r.vertices.length];

        let edge = VectorMath2.subtract(vb, va);
        axis = edge.perpendicular().normalise();

        var projA = projectVertices(r.vertices, axis);
        var projB = projectCircle(c, axis);

        if (projA.min >= projB.max || projB.min >= projA.max) {
            return null;
        }

        overlap = Math.min(
            projA.max - projB.min,
            projB.max - projA.min
        );

        if (overlap < depth) {
            depth = overlap;
            normal = axis;
        }
    }

    var minDist = Number.MAX_VALUE;
    var nearest = VectorMath2.zero();

    for (let i = 0; i < r.vertices.length; i++) {
        let v = r.vertices[i];
        let dist = VectorMath2.distance(c.pos, v);

        if (dist < minDist) {
            minDist = dist;
            nearest = v;
        }
    }// find the nearest point on the rectangle.
    //That distance vector is actually another axis to detect.

    axis = VectorMath2.subtract(nearest, c.pos).normalise();
    projA = projectVertices(r.vertices, axis);
    projB = projectCircle(c, axis);

    if (projA.min >= projB.max || projB.min >= projA.max) {
        return null;
    }

    overlap = Math.min(
        projA.max - projB.min,
        projB.max - projA.min
    );

    if (overlap < depth) {
        depth = overlap;
        normal = axis;
    }

    var dir = VectorMath2.subtract(r.pos, c.pos);
    if (dir.dot(normal) < 0) normal = normal.times(-1);

    //console.log("Circle hits rectangle!", normal);
    //error;

    return new Collision(
        c,
        r,
        normal,
        depth,
        c.pos.add(normal.times(c.radius))
    );
}

export function detectPolygons(p1, p2) {
    if (p1.mass === Infinity && p2.mass === Infinity) {
        return;
    }

    //console.log("polygons!",p1,p2);

    //console.log(p1,p2);

    var normal = new Vector2(0, 0);
    var depth = Number.MAX_VALUE;

    function checkCollision(pa, pb) {
        for (let i = 0; i < pa.vertices.length; i++) {
            //console.log(pa.vertices[(i + 1) % pa.vertices.length]);

            let va = pa.vertices[i];
            let vb = pa.vertices[(i + 1) % pa.vertices.length];

            //console.log("ra", ra);

            let edge = vb.subtract(va);
            let axis = edge.perpendicular().normalise();

            // get a seperating axis, pointing inward.

            let projA = projectVertices(pa.vertices, axis);
            let projB = projectVertices(pb.vertices, axis);

            if (projA.min >= projB.max || projB.min >= projA.max) {
                return false;
                // one axis can seperate 2 objects, no collision, return.
            }

            let overlap = Math.min(projA.max - projB.min, projB.max - projA.min);

            //console.log(depth,overlap);
            if (overlap < depth) {
                depth = overlap; // penetration is the minimum overlap
                normal = axis.clone();
                //console.log(normal);
            }
        }
        return true;
    }

    // Based on Seperating Axis Theorem
    // Only work for convex shapes
    // O(edge^2) time

    if (!checkCollision(p1, p2) || !checkCollision(p2, p1)) return null;

    //if(normal.x==0&&normal.y==0) return null;

    var dir = p2.pos.subtract(p1.pos);
    if (dir.dot(normal) < 0) normal = normal.times(-1);

    // if (normal.length() === 0) {
    //     console.error("Normal is zero before final adjustment");
    //     error;
    //     return null;
    // }

    var contactPoints1 = [];
    var contactPoints2 = [];
    //var contactPointA,contactPointB;
    var contactPoint = p1.pos.clone();

    var eps = 1e-5;

    for (let v of p1.vertices) {
        let diff = v.dot(normal) - contactPoint.dot(normal);

        if (diff >= -eps) {
            //console.log(v.dot(normal),contactPoint.dot(normal));
            if (diff > eps) {
                contactPoints1 = [];
            }
            contactPoint = v.clone();
            contactPoints1.push(contactPoint);
        }
    }

    contactPoint = p2.pos.clone();
    normal.timesEqual(-1);
    for (let v of p2.vertices) {
        let diff = v.dot(normal) - contactPoint.dot(normal);

        if (diff >= -eps) {
            //console.log(v.dot(normal),contactPoint.dot(normal));
            if (diff > eps) {
                contactPoints2 = [];
            }
            contactPoint = v.clone();
            contactPoints2.push(contactPoint);
        }
    }

    normal.timesEqual(-1);
    contactPoint = VectorMath2.zero();

    if (contactPoints1.length == 1) {
        contactPoint = contactPoints1[0];
    }
    else if (contactPoints2.length == 1) {
        contactPoint = contactPoints2[0];
    }
    else {
        //console.log(normal, contactPoints1.length, contactPoints2.length);

        var min1 = Number.MAX_VALUE;
        var min1v;
        var max1 = -Number.MAX_VALUE;
        var max1v;

        for (let c of contactPoints1) {
            let proj = c.dot(normal.perpendicular());
            if (proj < min1) {
                min1 = proj;
                min1v = c;
            }
            if (proj > max1) {
                max1 = proj;
                max1v = c;
            }
        }

        var min2 = Number.MAX_VALUE;
        var min2v;
        var max2 = -Number.MAX_VALUE;
        var max2v;
        for (let c of contactPoints2) {
            let proj = c.dot(normal.perpendicular());
            if (proj < min2) {
                min2 = proj;
                min2v = c;
            }
            if (proj > max2) {
                max2 = proj;
                max2v = c;
            }
        }

        // Draw.drawCircle(max2v, 0.01, "#0000FF");
        // Draw.drawCircle(min2v, 0.01, "#0000FF");
        // Draw.drawCircle(max1v, 0.01, "#00FFFF");
        // Draw.drawCircle(min1v, 0.01, "#00FFFF");

        //console.log(max2v);

        var contactLine = [max2 - min1, max1 - min2, max1 - min1, max2 - min2];
        
        if (!max2v || !min1v || !max2v || !min1v)
            return null;

        var contactPoints = [
            VectorMath2.add(max2v, min1v).times(1 / 2),
            VectorMath2.add(max1v, min2v).times(1 / 2),
            VectorMath2.add(max1v, min1v).times(1 / 2),
            VectorMath2.add(max2v, min2v).times(1 / 2),
        ];

        var minCL = Number.MAX_VALUE;
        for (let i = 0; i < contactLine.length; i++) {
            if (contactLine[i] < minCL) {
                minCL = contactLine[i];
                contactPoint = contactPoints[i];
            }
        }

    }
    //Average contact point

    Draw.drawCircle(contactPoint, 0.01, "#FFFF00");
    Draw.drawCircle(contactPoint.add(normal).times(0.1), 0.01, "FF0000");

    // if (normal.length() === 0) {
    //     console.error("Normal is zero");
    //     error;
    //     return null;
    // }

    return new Collision(
        p1,
        p2,
        normal,
        depth,
        contactPoint
    );
}

/**
 * @param {Vector2[]} vertices
 * @param {Vector2} axis
 */
function projectVertices(vertices, axis) {
    var min = Number.MAX_VALUE;
    var max = -Number.MAX_VALUE;
    for (let v of vertices) {
        let proj = v.dot(axis);
        min = Math.min(min, proj);
        max = Math.max(max, proj);
    }
    return { min: min, max: max };
}

/**
 * @param {Entity.Circle} c 
 * @param {Vector2} axis 
 */
function projectCircle(c, axis) {
    var p1 = c.pos.add(axis, c.radius);
    var p2 = c.pos.add(axis, -c.radius);

    var res1 = VectorMath2.dot(p1, axis);
    var res2 = VectorMath2.dot(p2, axis);

    return { min: Math.min(res1, res2), max: Math.max(res1, res2) };
}